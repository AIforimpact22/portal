# pages/2_Edit_Database.py
import time
from typing import List, Tuple

import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="Edit / Inspect Database", layout="wide")
st.title("Edit / Inspect Database")

db = DatabaseManager()  # Cloud SQL via pg8000

# ───────── Schema overview ─────────
st.subheader("Schema overview")

@st.cache_data(show_spinner=False, ttl=60)
def get_available_schemas() -> List[str]:
    q = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog','information_schema')
        ORDER BY schema_name;
    """
    df = db.fetch_data(q)
    return df["schema_name"].tolist() if not df.empty else ["public"]

@st.cache_data(show_spinner=False, ttl=60)
def get_schema_columns(schema: str) -> pd.DataFrame:
    q = """
        SELECT table_name, column_name, data_type
        FROM   information_schema.columns
        WHERE  table_schema = %s
        ORDER  BY table_name, ordinal_position;
    """
    return db.fetch_data(q, (schema,))

schemas = get_available_schemas()
schema = st.selectbox(
    "Schema",
    schemas or ["public"],
    index=(schemas or ["public"]).index("public") if "public" in (schemas or []) else 0,
)

schema_rows = get_schema_columns(schema)
if not schema_rows.empty:
    by_table: dict[str, list[str]] = {}
    for _, r in schema_rows.iterrows():
        t, c, d = r["table_name"], r["column_name"], r["data_type"]
        by_table.setdefault(t, []).append(f"{c} ({d})")
    for t, cols in by_table.items():
        st.markdown(f"**{t}**")
        st.write(", ".join(cols))
else:
    st.info("No tables found in this schema.")

st.caption(
    "Tip: Use single-transaction mode to run several statements atomically, "
    "or autocommit to run each statement independently."
)

# ───────── Helpers ─────────
def split_sql_statements(sql_text: str) -> List[str]:
    """Split SQL into statements (uses sqlparse if available; fallback otherwise)."""
    try:
        import sqlparse  # type: ignore
        stmts = [
            str(stmt).strip()
            for stmt in sqlparse.parse(sql_text or "")
            if str(stmt).strip().strip(";")
        ]
        return [s.rstrip(";").strip() for s in stmts if s.rstrip(";").strip()]
    except Exception:
        s = sql_text or ""
        out, buf = [], []
        in_single = in_double = in_dollar = False
        dollar_tag = ""
        i = 0
        while i < len(s):
            ch = s[i]
            buf.append(ch)

            if not in_single and not in_double:
                if not in_dollar and ch == "$":
                    j = i + 1
                    while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                        j += 1
                    if j < len(s) and s[j] == "$":
                        in_dollar = True
                        dollar_tag = s[i : j + 1]  # like $tag$
                        i = j
                elif in_dollar and ch == "$":
                    tag_len = len(dollar_tag)
                    if i - tag_len + 1 >= 0 and s[i - tag_len + 1 : i + 1] == dollar_tag:
                        in_dollar = False
                        dollar_tag = ""

            if not in_dollar:
                if ch == "'" and not in_double and (i == 0 or s[i - 1] != "\\"):
                    in_single = not in_single
                elif ch == '"' and not in_single and (i == 0 or s[i - 1] != "\\"):
                    in_double = not in_double
                elif ch == ";" and not in_single and not in_double:
                    stmt = "".join(buf).strip()[:-1].strip()
                    if stmt:
                        out.append(stmt)
                    buf = []
            i += 1

        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
        return [x for x in out if x]

def _set_timeouts(cur, lock_timeout_ms: int, statement_timeout_ms: int, in_txn: bool):
    if in_txn:
        cur.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms';")
        cur.execute(f"SET LOCAL statement_timeout = '{int(statement_timeout_ms)}ms';")
        cur.execute("SET LOCAL idle_in_transaction_session_timeout = '15000ms';")
        return False
    else:
        cur.execute(f"SET lock_timeout = '{int(lock_timeout_ms)}ms';")
        cur.execute(f"SET statement_timeout = '{int(statement_timeout_ms)}ms';")
        return True

def _reset_timeouts(cur):
    cur.execute("RESET lock_timeout;")
    cur.execute("RESET statement_timeout;")


def _safe_rollback():
    try:
        db.conn.rollback()
    except Exception:
        try:
            db._reconnect()
        except Exception:
            pass


def _pg8000_err_to_text(e: Exception) -> str:
    try:
        first = e.args[0]
        if isinstance(first, dict):
            code = first.get("C", "")
            detail = first.get("D", "")
            msgtxt = first.get("M", str(e))
            name = first.get("n", "")
            rel = first.get("t", "")
            return f"[{code}] {msgtxt}" + (f" • {detail}" if detail else "") + (f" • table={rel}" if rel else "") + (f" • constraint={name}" if name else "")
    except Exception:
        pass
    return str(e)

def run_one_statement(
    sql: str,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
    max_rows: int | None,
    explain: bool,
    in_txn: bool,
) -> Tuple[str, object, float]:
    """
    Returns: (kind, payload, elapsed_sec)
      kind="result" -> payload is DataFrame
      kind="ok"     -> payload is status string (built from rowcount)
    """
    to_exec = sql if not explain else f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {sql}"

    start = time.perf_counter()
    cur = db.conn.cursor()
    needs_reset = False
    try:
        needs_reset = _set_timeouts(cur, lock_timeout_ms, statement_timeout_ms, in_txn)
        cur.execute(to_exec)

        if cur.description:  # query returned rows
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            if max_rows is not None and max_rows > 0 and len(rows) > max_rows:
                rows = rows[:max_rows]
            payload = pd.DataFrame(rows, columns=cols)
            kind = "result"
        else:
            rc = cur.rowcount
            # pg8000 cursor has no statusmessage; craft a friendly message
            if rc in (-1, None):
                payload = "Command executed."
            else:
                payload = f"Command executed • rowcount={rc}"
            kind = "ok"

        elapsed = time.perf_counter() - start
        return (kind, payload, elapsed)
    finally:
        try:
            if needs_reset:
                _reset_timeouts(cur)
        except Exception:
            pass
        cur.close()

# ───────── SQL Runner UI ─────────
st.subheader("Run arbitrary SQL")

with st.expander("Execution settings", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        single_txn = st.checkbox(
            "Run all statements in a single transaction (rollback on first error)",
            value=False,
            help="If unchecked, each statement runs & commits independently (autocommit).",
        )
    with col2:
        lock_timeout_ms = st.number_input(
            "Lock timeout (ms)", min_value=0, max_value=60000, value=2000, step=250,
            help="Fail fast if another session holds a conflicting lock.",
        )
    with col3:
        statement_timeout_ms = st.number_input(
            "Statement timeout (ms)", min_value=0, max_value=10_000_000, value=30_000, step=500,
            help="Abort statements that run too long.",
        )
    col4, col5 = st.columns(2)
    with col4:
        max_rows = st.number_input(
            "Max rows to display", min_value=100, max_value=1_000_000, value=5000, step=100,
            help="Limits result-set size shown in the UI.",
        )
    with col5:
        explain = st.checkbox(
            "EXPLAIN ANALYZE (read-only)",
            value=False,
            help="Prepends EXPLAIN (ANALYZE, BUFFERS, VERBOSE) to each statement.",
        )

sql_txt = st.text_area(
    "SQL to execute inside this DB (multiple statements supported)",
    height=220,
    placeholder=(
        "Paste one or more SQL statements separated by semicolons…\n\n"
        "Example:\n"
        "CREATE TABLE public.shelf_map_locations (\n"
        "  locid varchar NOT NULL,\n"
        "  label varchar,\n"
        "  x_pct numeric,\n"
        "  y_pct numeric,\n"
        "  w_pct numeric,\n"
        "  h_pct numeric,\n"
        "  rotation_deg real,\n"
        "  PRIMARY KEY (locid)\n"
        ");\n"
        "SELECT * FROM information_schema.tables WHERE table_schema='public';"
    ),
)

if st.button("Run SQL"):
    stmts = split_sql_statements(sql_txt or "")
    if not stmts:
        st.warning("No SQL statements detected.")
        st.stop()

    with st.spinner(f"Executing {len(stmts)} statement(s)…"):
        if single_txn:
            # Run all in one transaction
            cur = db.conn.cursor()
            try:
                cur.execute("BEGIN;")
                cur.close()
                for idx, stmt in enumerate(stmts, start=1):
                    pretty = f"Statement {idx}"
                    with st.container(border=True):
                        st.markdown(f"**{pretty}**")
                        st.code(stmt, language="sql")
                        try:
                            kind, payload, elapsed = run_one_statement(
                                stmt,
                                lock_timeout_ms,
                                statement_timeout_ms,
                                int(max_rows),
                                explain,
                                in_txn=True,
                            )
                        except Exception as e:
                            _safe_rollback()
                            st.error(_pg8000_err_to_text(e))
                            st.warning("Transaction rolled back due to the error above.")
                            st.stop()

                        if kind == "result":
                            st.dataframe(payload, use_container_width=True)
                            st.caption(f"Returned {len(payload)} row(s) • {elapsed*1000:.0f} ms")
                        else:
                            st.success(f"{payload} • {elapsed*1000:.0f} ms")
                db.conn.commit()
            finally:
                try:
                    cur = db.conn.cursor()
                    cur.execute("END;")
                    cur.close()
                except Exception:
                    pass
        else:
            # Autocommit per statement
            for idx, stmt in enumerate(stmts, start=1):
                pretty = f"Statement {idx}"
                with st.container(border=True):
                    st.markdown(f"**{pretty}**")
                    st.code(stmt, language="sql")
                    try:
                        kind, payload, elapsed = run_one_statement(
                            stmt,
                            lock_timeout_ms,
                            statement_timeout_ms,
                            int(max_rows),
                            explain,
                            in_txn=False,
                        )
                        db.conn.commit()
                    except Exception as e:
                        _safe_rollback()
                        st.error(_pg8000_err_to_text(e))
                        continue

                    if kind == "result":
                        st.dataframe(payload, use_container_width=True)
                        st.caption(f"Returned {len(payload)} row(s) • {elapsed*1000:.0f} ms")
                    else:
                        st.success(f"{payload} • {elapsed*1000:.0f} ms")
