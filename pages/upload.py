# pages/upload.py
import io
import math
import time
import json
import unicodedata
import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="CSV → Table Upload (NULL fix + Multi-row INSERT)", layout="wide")
st.title("⬆️ CSV → Table Upload (NULL token normalization + Upsert + Debug)")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Debug log (session)
# ───────────────────────────────────────────────────────────────
if "upload_logs" not in st.session_state:
    st.session_state.upload_logs = []

def log(msg: str, data: dict | None = None):
    st.session_state.upload_logs.append({
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "msg": msg,
        **({"data": data} if data else {})
    })

def dump_logs_text() -> str:
    lines = []
    for e in st.session_state.upload_logs:
        if "data" in e:
            lines.append(f'{e["t"]} | {e["msg"]} | {json.dumps(e["data"], ensure_ascii=False)}')
        else:
            lines.append(f'{e["t"]} | {e["msg"]}')
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────────
# Catalog helpers
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def list_tables() -> pd.DataFrame:
    q = """
      SELECT table_schema, table_name
      FROM information_schema.tables
      WHERE table_schema NOT IN ('pg_catalog','information_schema')
        AND table_type='BASE TABLE'
      ORDER BY table_schema, table_name
    """
    df = db.fetch_data(q)
    return df if not df.empty else pd.DataFrame(columns=["table_schema","table_name"])

@st.cache_data(show_spinner=False, ttl=60)
def table_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
      SELECT
        c.ordinal_position,
        c.column_name,
        c.data_type,
        c.is_nullable = 'YES' AS is_nullable,
        c.column_default,
        c.character_maximum_length,
        c.numeric_precision,
        c.numeric_scale
      FROM information_schema.columns c
      WHERE c.table_schema=%s AND c.table_name=%s
      ORDER BY c.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df.empty:
        return pd.DataFrame(columns=[
            "ordinal_position","column_name","data_type","is_nullable",
            "column_default","character_maximum_length","numeric_precision","numeric_scale"
        ])
    return df

@st.cache_data(show_spinner=False, ttl=60)
def primary_keys(schema: str, table: str) -> list[str]:
    q = """
      SELECT kcu.column_name
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema   = kcu.table_schema
      WHERE tc.constraint_type='PRIMARY KEY'
        AND tc.table_schema=%s AND tc.table_name=%s
      ORDER BY kcu.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    return df["column_name"].tolist() if not df.empty and "column_name" in df.columns else []

@st.cache_data(show_spinner=False, ttl=60)
def unique_constraints(schema: str, table: str) -> list[list[str]]:
    """
    Return a list of UNIQUE constraint column lists, including PK.
    """
    q = """
      SELECT c.conname,
             ARRAY_AGG(a.attname ORDER BY a.attnum) AS cols
      FROM   pg_constraint c
      JOIN   pg_class t  ON t.oid = c.conrelid
      JOIN   pg_namespace n ON n.oid = t.relnamespace
      JOIN   unnest(c.conkey) WITH ORDINALITY AS ck(attnum, ord) ON TRUE
      JOIN   pg_attribute a ON a.attrelid = t.oid AND a.attnum = ck.attnum
      WHERE  n.nspname=%s AND t.relname=%s AND c.contype IN ('p','u')
      GROUP BY c.conname
      ORDER BY c.conname
    """
    df = db.fetch_data(q, (schema, table))
    if df.empty or "cols" not in df.columns:
        return []
    return [list(v) for v in df["cols"].tolist()]

# ───────────────────────────────────────────────────────────────
# CSV + mapping helpers
# ───────────────────────────────────────────────────────────────
def normalize_name(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    return s.lower().replace(" ", "_")

def automap(csv_cols, table_cols):
    norm = {normalize_name(c): c for c in csv_cols}
    return {t: norm.get(normalize_name(t)) for t in table_cols}

def read_csv(uploaded, delimiter, encoding, has_header, quotechar):
    raw = uploaded.read()
    opts = dict(sep=delimiter or ",", encoding=encoding or "utf-8", dtype=str,
                keep_default_na=False, na_values=[""], quotechar=quotechar or '"')
    if has_header:
        return pd.read_csv(io.BytesIO(raw), **opts)
    df = pd.read_csv(io.BytesIO(raw), header=None, **opts)
    df.columns = [f"column_{i+1}" for i in range(df.shape[1])]
    return df

def example_for_type(dt: str) -> str:
    dt = (dt or "").lower()
    if "int" in dt: return "123"
    if "numeric" in dt or "decimal" in dt: return "9.99"
    if "double" in dt or "real" in dt or "float" in dt: return "3.14"
    if "bool" in dt: return "true"
    if dt=="date": return "2025-01-01"
    if "timestamp" in dt: return "2025-01-01 12:34:56"
    if dt=="time": return "12:34:56"
    if "uuid" in dt: return "00000000-0000-0000-0000-000000000000"
    if "json" in dt: return '{"key":"value"}'
    if "char" in dt or "text" in dt: return "example"
    return "value"

def csv_template(cols_df: pd.DataFrame) -> pd.DataFrame:
    if cols_df.empty:
        return pd.DataFrame([{"example_column":"value"},{"example_column":"value"}])
    row={}
    for _,r in cols_df.iterrows():
        row[r["column_name"]] = example_for_type(r["data_type"])
    return pd.DataFrame([row,row])

def required_columns(cols_df: pd.DataFrame, pks: list[str]) -> list[str]:
    req=[]
    if cols_df.empty: return req
    for _,r in cols_df.iterrows():
        c=r["column_name"]
        not_null = not bool(r.get("is_nullable", False))
        has_def  = str(r.get("column_default","") or "")!=""
        if c in pks and not has_def: req.append(c)
        elif not_null and not has_def: req.append(c)
    return req

# ───────────────────────────────────────────────────────────────
# SQL builder (multi-row VALUES)
# ───────────────────────────────────────────────────────────────
def build_multirow_insert_sql(schema: str,
                              table: str,
                              cols: list[str],
                              n_rows: int,
                              mode: str,
                              conflict_target: tuple[str, ...] | None) -> str:
    cols_quoted = ", ".join([f'"{c}"' for c in cols])
    row_ph = "(" + ", ".join(["%s"] * len(cols)) + ")"
    values_clause = ", ".join([row_ph] * n_rows)
    sql = f'INSERT INTO "{schema}"."{table}" ({cols_quoted}) VALUES {values_clause}'
    if mode == "skip":
        sql += " ON CONFLICT DO NOTHING"
    elif mode == "upsert" and conflict_target:
        conflict_cols = ", ".join([f'"{c}"' for c in conflict_target])
        update_cols = [c for c in cols if c not in conflict_target]
        if update_cols:
            set_clause = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in update_cols])
            sql += f" ON CONFLICT ({conflict_cols}) DO UPDATE SET {set_clause}"
        else:
            sql += " ON CONFLICT DO NOTHING"
    return sql

def flatten_params(rows: list[tuple]) -> list:
    flat=[]
    for r in rows: flat.extend(r)
    return flat

# ───────────────────────────────────────────────────────────────
# UI controls
# ───────────────────────────────────────────────────────────────
cA, cB, cC = st.columns([1.6, 1.2, 1])
with cA:
    delim = st.text_input("Delimiter", value=",", help="e.g. , ; | \\t").replace("\\t","\t")
    enc = st.text_input("Encoding", value="utf-8")
with cB:
    quotechar = st.text_input("Quote char", value='"')
    has_header = st.checkbox("CSV has header row", value=True)
    # NULL token normalization settings
    st.markdown("**NULL token normalization**")
    null_tokens_text = st.text_input(
        "Treat these tokens as NULL (case-insensitive, comma-separated)",
        value="None,null,nan,na,n/a",
        help="These string values will be converted to SQL NULL."
    )
    NULL_TOKENS = {t.strip().lower() for t in null_tokens_text.split(",") if t.strip() != ""}
with cC:
    rows_per_stmt = st.number_input("Rows per INSERT statement", min_value=200, max_value=5000, value=1000, step=100)
    progress_chunk = st.number_input("Progress update step", min_value=rows_per_stmt, max_value=50000, value=2000, step=rows_per_stmt)

tables = list_tables()
schemas = sorted(tables["table_schema"].unique().tolist())
schema = st.selectbox("Schema", options=schemas, index=(schemas.index("public") if "public" in schemas else 0))
subset = tables[tables["table_schema"]==schema]
table  = st.selectbox("Table", options=subset["table_name"].tolist() or ["— none —"])

with st.expander("🐞 Debug panel", expanded=False):
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain", key="dbg_dl_top")

if not table or table=="— none —":
    st.info("Pick a schema & table.")
    st.stop()

cols_df = table_columns(schema, table)
pks     = primary_keys(schema, table)
uniqs   = unique_constraints(schema, table)
tmpl    = csv_template(cols_df)

st.subheader("📘 Data dictionary")
dd = cols_df.copy()
if not dd.empty:
    dd.insert(1,"is_primary_key", dd["column_name"].isin(pks))
st.dataframe(dd if not dd.empty else pd.DataFrame([{"info":"No columns found."}]),
             use_container_width=True, hide_index=True)

with st.expander("📄 CSV Template & Examples", expanded=True):
    st.write("**Template preview (example values):**")
    st.dataframe(tmpl, use_container_width=True)
    st.download_button("⬇️ Download CSV template",
                       data=tmpl.to_csv(index=False).encode("utf-8"),
                       file_name=f"{schema}.{table}.template.csv",
                       mime="text/csv",
                       key="tmpl_dl_top")

uploaded = st.file_uploader("Choose CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to continue.")
    st.stop()

# Read CSV
try:
    df_csv = read_csv(uploaded, delim, enc, has_header, quotechar)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    log("read_csv_failed", {"error": str(e)})
    st.stop()

st.write("**CSV Preview**")
st.dataframe(df_csv.head(50), use_container_width=True)

# Mapping UI
st.subheader("Map CSV columns → table columns")
table_cols = cols_df["column_name"].tolist() if "column_name" in cols_df.columns else []
if not table_cols:
    st.error("No table columns; cannot continue.")
    st.stop()

default_map = automap(df_csv.columns.tolist(), table_cols)
mapping = {}
m1, m2 = st.columns([2, 2])
with m1:
    st.caption("Table column → CSV column")
    for tcol in table_cols:
        default_choice = default_map.get(tcol)
        mapping[tcol] = st.selectbox(
            f'↪ {tcol}',
            options=["— skip —"] + df_csv.columns.tolist(),
            index=(["— skip —"] + df_csv.columns.tolist()).index(default_choice) if default_choice in df_csv.columns else 0,
            key=f"map_{tcol}"
        )
with m2:
    truncate = st.checkbox("TRUNCATE table before load (danger!)", value=False)
    mode_opt = st.radio("On conflict", ["Skip (DO NOTHING)", "Upsert (DO UPDATE)"], index=0)
    mode = "skip" if mode_opt.startswith("Skip") else "upsert"

    # Conflict targets from PK + unique constraints
    conflict_options = []
    if pks: conflict_options.append(tuple(pks))
    for combo in uniqs:
        tup = tuple(combo)
        if tup and tup not in conflict_options:
            conflict_options.append(tup)
    conflict_target = st.selectbox(
        "Conflict target (unique key)",
        options=conflict_options or [()],
        format_func=lambda x: ", ".join(x) if x else "(none)",
    )

# Determine target columns and required validation
def required_columns_local(cols_df: pd.DataFrame, pks: list[str]) -> list[str]:
    req=[]
    if cols_df.empty: return req
    for _,r in cols_df.iterrows():
        c=r["column_name"]
        not_null = not bool(r.get("is_nullable", False))
        has_def  = str(r.get("column_default","") or "")!=""
        if c in pks and not has_def: req.append(c)
        elif not_null and not has_def: req.append(c)
    return req

target_cols = [c for c in table_cols if mapping.get(c) and mapping[c] != "— skip —"]
req_cols = required_columns_local(cols_df, pks)
missing_req = [c for c in req_cols if c not in target_cols]
if missing_req:
    st.error(f"Required columns not mapped: {', '.join(missing_req)}")
    st.stop()
if not target_cols:
    st.warning("Map at least one column.")
    st.stop()

# Build mapped DataFrame (apply NULL token normalization)
def to_null_if_token(val: str):
    if val is None:
        return None
    if isinstance(val, str):
        v = val.strip()
        # empty string -> NULL
        if v == "":
            return None
        # token set (case-insensitive)
        if v.lower() in NULL_TOKENS:
            return None
        return v
    return val

mapped = pd.DataFrame()
for tcol in target_cols:
    mapped[tcol] = df_csv[mapping[tcol]].map(to_null_if_token)

# Preflight: show how many NULLs created per column; warn for unique cols
st.subheader("Preflight checks")
null_counts = mapped.isna().sum()
st.write("**Null counts after token normalization:**")
st.dataframe(null_counts.to_frame("null_count"), use_container_width=True)

# If a unique/PK column will be mostly NULL → warn (but Postgres allows multiple NULLs)
unique_cols = set()
for combo in ([tuple(pks)] if pks else []):
    unique_cols.update(combo)
for combo in unique_constraints(schema, table):
    unique_cols.update(combo)
uc_warn = {c: int(null_counts.get(c, 0)) for c in unique_cols if c in null_counts and null_counts[c] > 0}
if uc_warn:
    st.info(f"Note: Unique columns with NULLs (allowed in Postgres): {uc_warn}")

st.write("**Mapped Preview (post-normalization):**")
st.dataframe(mapped.head(50), use_container_width=True)
st.caption(f"{len(mapped):,} rows → {schema}.{table} columns {target_cols}")

# Optional row-by-row debug (first N)
row_debug = st.checkbox("Row-by-row debug (first N)", value=False,
                        help="Runs single-row INSERT for the first N rows to diagnose exact errors.")
row_debug_n = st.number_input("N", min_value=1, max_value=200, value=25, step=1, disabled=not row_debug)

def build_sql_for_n(n_rows: int) -> str:
    return build_multirow_insert_sql(
        schema, table, target_cols, n_rows,
        mode, (conflict_target if mode=="upsert" else None)
    )

if row_debug:
    st.info("Debugging first N rows individually…")
    sql_one = build_sql_for_n(1)
    results = []
    for idx, row in mapped.head(int(row_debug_n)).iterrows():
        params = tuple(row[c] for c in target_cols)
        cur = db.conn.cursor()
        try:
            cur.execute("SET LOCAL statement_timeout = 15000;")
            cur.execute(sql_one, params)
            db.conn.commit()
            results.append({"row": int(idx + 1), "status": "ok"})
        except Exception as e:
            db.conn.rollback()
            results.append({"row": int(idx + 1), "status": "error", "error": str(e)})
        finally:
            cur.close()
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    log("row_debug", {"results": results})

# ───────────────────────────────────────────────────────────────
# Execute upload (multi-row INSERT, no executemany)
# ───────────────────────────────────────────────────────────────
go = st.button("🚀 Start upload")
if not go:
    st.stop()

if truncate:
    st.error("You chose TRUNCATE (deletes ALL rows first).")
    if not st.checkbox("I understand, proceed with TRUNCATE"):
        st.stop()
    try:
        db.execute_command(f'TRUNCATE TABLE "{schema}"."{table}" RESTART IDENTITY CASCADE;')
        log("truncate_done", {"schema": schema, "table": table})
    except Exception as e:
        st.error(f"TRUNCATE failed: {e}")
        log("truncate_failed", {"error": str(e)})
        st.stop()

total_rows = len(mapped)
rows_per_stmt = int(rows_per_stmt)
num_stmts = int(math.ceil(total_rows / rows_per_stmt)) if total_rows else 0

log("upload_start", {
    "schema": schema, "table": table,
    "rows": total_rows,
    "rows_per_stmt": rows_per_stmt,
    "statements": num_stmts,
    "mode": mode,
    "conflict_target": list(conflict_target) if conflict_target else None
})

prog = st.progress(0.0)
status = st.empty()
t_all = time.time()
attempted_total = 0

try:
    for i in range(num_stmts):
        start_idx = i * rows_per_stmt
        end_idx   = min((i + 1) * rows_per_stmt, total_rows)
        batch = mapped.iloc[start_idx:end_idx]

        rows_tuples = list(batch[target_cols].itertuples(index=False, name=None))
        sql_stmt = build_sql_for_n(len(rows_tuples))
        params = flatten_params(rows_tuples)

        cur = db.conn.cursor()
        t0 = time.time()
        try:
            cur.execute("SET LOCAL statement_timeout = 180000;")
            cur.execute(sql_stmt, params)
        except Exception as e:
            db.conn.rollback()
            cur.close()
            log("stmt_failed", {
                "stmt": i + 1,
                "rows_in_stmt": len(rows_tuples),
                "range": [start_idx + 1, end_idx],
                "error": str(e)
            })
            st.error(f"Statement {i+1}/{num_stmts} failed (rows {start_idx+1}-{end_idx}). {e}")
            st.stop()
        else:
            db.conn.commit()
            cur.close()
            elapsed = time.time() - t0
            rps = len(rows_tuples) / elapsed if elapsed > 0 else None
            attempted_total += len(rows_tuples)
            log("stmt_ok", {
                "stmt": i + 1,
                "rows_in_stmt": len(rows_tuples),
                "range": [start_idx + 1, end_idx],
                "elapsed_sec": round(elapsed, 3),
                "rows_per_sec": round(rps, 1) if rps else None
            })

        # progress UI
        if (attempted_total % int(progress_chunk) == 0) or (i == num_stmts - 1):
            prog.progress(attempted_total / max(total_rows, 1))
            status.info(f"Attempted {attempted_total:,}/{total_rows:,} rows "
                        f"(statement {i+1}/{num_stmts}).")

    total_elapsed = time.time() - t_all
    rps_total = attempted_total / total_elapsed if total_elapsed > 0 else None
    log("upload_done", {
        "attempted_total": attempted_total,
        "elapsed_sec": round(total_elapsed, 2),
        "rows_per_sec": round(rps_total or 0, 1)
    })

    prog.progress(1.0)
    st.success(f"Done! Attempted {attempted_total:,} rows in {total_elapsed:.2f}s "
               f"({(rps_total or 0):.0f} rows/sec).")
    st.balloons()

except Exception as e:
    log("upload_exception", {"error": str(e)})
    st.error(f"Upload failed: {e}")

with st.expander("🐞 Debug panel", expanded=False):
    st.code(dump_logs_text() or "No logs yet.", language="text")
    st.download_button("⬇️ Download logs", data=(dump_logs_text() or "No logs."),
                       file_name="upload_debug_logs.txt", mime="text/plain", key="dbg_dl_bottom")
