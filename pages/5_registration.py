# pages/registrations_status.py
import os
import streamlit as st
import pandas as pd
from typing import List, Any, Optional, Tuple

from db_handler import DatabaseManager  # read-only

st.set_page_config(page_title="Registrations — Enrollment Status", layout="wide")
st.title("📝 Registrations — Enrollment Status (resilient writer)")

# ───────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────
TABLE = "registrations"
STATUS_FIELD = "enrollment_status"
ALLOWED_STATUSES = ["pending", "accepted", "rejected"]

db = DatabaseManager()  # reads only

# ───────────────────────────────────────────────────────────────
# WRITE LAYER — short-lived connection, multiple fallbacks
# ───────────────────────────────────────────────────────────────
def _env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip()

def _get_basic_creds():
    # prefer DB_* then PG_*
    user = _env("DB_USER") or _env("PGUSER")
    password = _env("DB_PASS") or _env("PGPASSWORD")
    database = _env("DB_NAME") or _env("PGDATABASE")
    host = _env("DB_HOST") or _env("PGHOST") or "localhost"
    port = int(_env("DB_PORT") or _env("PGPORT") or "5432")
    return user, password, database, host, port

def _exec_write(sql: str, params: Tuple[Any, ...]) -> int:
    """
    Run a write and return affected row count using a tiny SELECT wrapper.
    Tries:
      1) psycopg2 with DATABASE_URL (if set)
      2) pg8000 with basic creds (DB_* / PG_*)
      3) Cloud SQL Connector (INSTANCE_CONNECTION_NAME + DB_USER/DB_PASS/DB_NAME)
    """
    # 1) psycopg2 with DATABASE_URL
    db_url = _env("DATABASE_URL") or _env("DB_URL")
    if db_url:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn = psycopg2.connect(db_url, connect_timeout=10)
            conn.autocommit = True
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    return int(row["affected"]) if row and "affected" in row else 0
            finally:
                conn.close()
        except Exception:
            pass  # fall through

    # 2) pg8000 with basic creds
    try:
        import pg8000.native as pg
        user, password, database, host, port = _get_basic_creds()
        if user and database:
            conn = pg.Connection(
                user=user,
                password=password,
                host=host,
                port=port,
                database=database,
                timeout=10,
            )
            try:
                res = conn.run(sql, params)  # returns list of tuples
                if res and len(res[0]) >= 1:
                    return int(res[0][0])
                return 0
            finally:
                conn.close()
    except Exception:
        pass  # fall through

    # 3) Cloud SQL Connector (pg8000)
    try:
        from google.cloud.sql.connector import Connector
        import pg8000  # dbapi variant used by connector
        instance = _env("INSTANCE_CONNECTION_NAME")
        user = _env("DB_USER") or _env("PGUSER")
        password = _env("DB_PASS") or _env("PGPASSWORD")
        database = _env("DB_NAME") or _env("PGDATABASE")
        if instance and user and database is not None:
            with Connector() as connector:
                conn = connector.connect(
                    instance,
                    "pg8000",
                    user=user,
                    password=password,
                    db=database,
                )
                try:
                    # dbapi cursor
                    cur = conn.cursor()
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    cur.close()
                    # some drivers return tuple, first column is affected
                    if row is None:
                        return 0
                    if isinstance(row, tuple):
                        return int(row[0])
                    # dict-like fallback
                    return int(row.get("affected", 0))
                finally:
                    conn.close()
    except Exception as e:
        # Surface the final error with a clear hint of what's missing
        missing = []
        if not db_url:
            # only complain about creds if URL not provided
            u, p, d, h, po = _get_basic_creds()
            if not u: missing.append("DB_USER/PGUSER")
            if d is None: missing.append("DB_NAME/PGDATABASE")
        icn = _env("INSTANCE_CONNECTION_NAME")
        if not db_url and not icn:
            missing.append("INSTANCE_CONNECTION_NAME (for Cloud SQL)")
        hint = f"Missing config: {', '.join(missing)}" if missing else str(e)
        raise RuntimeError(f"Write failed (no driver usable): {hint}")

    # If all 3 paths failed silently (shouldn't), raise a clear error
    raise RuntimeError("Write failed: no usable connection method (DATABASE_URL / DB_* or Cloud SQL).")

# Always wrap UPDATE so we SELECT an 'affected' count
def write_update_single(schema: str, pk_col: str, pk_val: Any, new_status: str) -> int:
    sql = f'''
        WITH upd AS (
            UPDATE "{schema}"."{TABLE}"
               SET "{STATUS_FIELD}" = %s
             WHERE "{pk_col}" = %s
         RETURNING 1
        )
        SELECT COUNT(*)::int AS affected FROM upd
    '''
    return _exec_write(sql, (new_status, pk_val))

def write_update_ids(schema: str, pk_col: str, ids: List[Any], new_status: str) -> int:
    if not ids:
        return 0
    sql = f'''
        WITH upd AS (
            UPDATE "{schema}"."{TABLE}"
               SET "{STATUS_FIELD}" = %s
             WHERE "{pk_col}" = ANY(%s)
         RETURNING 1
        )
        SELECT COUNT(*)::int AS affected FROM upd
    '''
    return _exec_write(sql, (new_status, ids))

def write_update_filter(schema: str, from_status: str, to_status: str) -> int:
    sql = f'''
        WITH upd AS (
            UPDATE "{schema}"."{TABLE}"
               SET "{STATUS_FIELD}" = %s
             WHERE "{STATUS_FIELD}" = %s
         RETURNING 1
        )
        SELECT COUNT(*)::int AS affected FROM upd
    '''
    return _exec_write(sql, (to_status, from_status))

# ───────────────────────────────────────────────────────────────
# READ HELPERS (DatabaseManager)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def find_schema_for_table(table_name: str) -> str:
    q = """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_name = %s
          AND table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY 1
        LIMIT 1
    """
    df = db.fetch_data(q, (table_name,))
    return df["table_schema"].iat[0] if isinstance(df, pd.DataFrame) and not df.empty else "public"

@st.cache_data(show_spinner=False, ttl=120)
def get_primary_key(schema: str, table: str) -> List[str]:
    q = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
          AND tc.table_name = %s
        ORDER BY kcu.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    return df["column_name"].tolist() if isinstance(df, pd.DataFrame) and not df.empty else []

@st.cache_data(show_spinner=False, ttl=120)
def status_column_exists(schema: str, table: str, col: str) -> bool:
    q = """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        LIMIT 1
    """
    df = db.fetch_data(q, (schema, table, col))
    return isinstance(df, pd.DataFrame) and not df.empty

@st.cache_data(show_spinner=True, ttl=30)
def load_preview(schema: str, status: str, limit: int, pk_col: Optional[str]) -> pd.DataFrame:
    sql = f'SELECT * FROM "{schema}"."{TABLE}"'
    params: Tuple[Any, ...] | None = None
    if status != "all":
        sql += f' WHERE "{STATUS_FIELD}" = %s'
        params = (status,)
    sql += f' ORDER BY "{pk_col}"' if pk_col else " ORDER BY 1"
    sql += f' LIMIT {int(limit)}'
    df = db.fetch_data(sql, params)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=30)
def count_by_status(schema: str) -> pd.DataFrame:
    sql = f'''
        SELECT COALESCE("{STATUS_FIELD}"::text, 'NULL') AS status, COUNT(*)::bigint AS ct
        FROM "{schema}"."{TABLE}"
        GROUP BY 1 ORDER BY 1;
    '''
    df = db.fetch_data(sql)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame({"status": [], "ct": []})
    return df

# ───────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────
schema = find_schema_for_table(TABLE)
pk_cols = get_primary_key(schema, TABLE)
if len(pk_cols) != 1:
    st.error("This page requires a **single-column** primary key on 'registrations'.")
    if pk_cols:
        st.info(f"Detected PK columns: {', '.join(pk_cols)}")
    st.stop()
pk = pk_cols[0]

if not status_column_exists(schema, TABLE, STATUS_FIELD):
    st.error(f"Column '{STATUS_FIELD}' not found in {schema}.{TABLE}.")
    st.stop()

with st.expander("Overview", expanded=False):
    stats = count_by_status(schema)
    def _get(s: str) -> int:
        if stats.empty: return 0
        row = stats.loc[stats["status"] == s, "ct"]
        return int(row.iat[0]) if not row.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("pending", f"{_get('pending'):,}")
    c2.metric("accepted", f"{_get('accepted'):,}")
    c3.metric("rejected", f"{_get('rejected'):,}")

st.subheader("Browse & select")
col_f1, col_f2, col_f3 = st.columns([1.3, 1, 1])
with col_f1:
    status_choice = st.radio("Filter by status", ["pending", "accepted", "rejected", "all"], index=0, horizontal=True)
with col_f2:
    limit = st.number_input("Preview rows", 5, 2000, 100, step=25)
with col_f3:
    if st.button("🔄 Refresh", use_container_width=True):
        find_schema_for_table.clear(); get_primary_key.clear()
        status_column_exists.clear(); load_preview.clear(); count_by_status.clear()
        st.rerun()

df = load_preview(schema, status_choice, int(limit), pk)
if df.empty:
    st.info("No rows found for the current filter.")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download preview CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{schema}.{TABLE}.{status_choice}.csv",
    mime="text/csv",
)

# Safe, typed PK options (no FutureWarning)
pk_options: List[Any] = pd.Series(df[pk]).dropna().astype(object).unique().tolist()

# ───────────────────────────────────────────────────────────────
# Bulk update — selected
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bulk update — selected")

ids_selected: List[Any] = st.multiselect(
    f"Select {len(pk_options)} registration(s) by primary key",
    options=pk_options,
    format_func=lambda v: str(v),
)
new_status_selected = st.selectbox("Set selected rows to", ALLOWED_STATUSES, index=0)

if st.button("✅ Apply to selected", type="primary"):
    if not ids_selected:
        st.warning("Pick at least one registration.")
    else:
        try:
            affected = write_update_ids(schema, pk, ids_selected, new_status_selected)
            load_preview.clear(); count_by_status.clear()
            st.success(f"Updated {affected} row(s) to '{new_status_selected}'.")
            st.rerun()
        except Exception as e:
            st.error(f"Bulk update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Bulk update — all in preview / all in DB filter
# ───────────────────────────────────────────────────────────────
st.subheader("Bulk update — quick apply")

col_q1, col_q2 = st.columns(2)
with col_q1:
    target_status_preview = st.selectbox("Set **all rows in preview** to", ALLOWED_STATUSES, index=0, key="preview_apply")
    if st.button("⚡ Apply to all in preview"):
        try:
            ids_in_preview = df[pk].dropna().tolist()
            affected = write_update_ids(schema, pk, ids_in_preview, target_status_preview)
            load_preview.clear(); count_by_status.clear()
            st.success(f"Updated {affected} previewed row(s) to '{target_status_preview}'.")
            st.rerun()
        except Exception as e:
            st.error(f"Bulk (preview) update failed: {e}")

with col_q2:
    disabled = status_choice == "all"
    target_status_filter = st.selectbox("Set **all rows in current filter** to", ALLOWED_STATUSES, index=0, key="filter_apply", disabled=disabled)
    if st.button("⚡ Apply to all matching filter", disabled=disabled):
        if disabled:
            st.info("Choose a specific status filter (not 'all') to use this action.")
        else:
            try:
                affected = write_update_filter(schema, status_choice, target_status_filter)
                load_preview.clear(); count_by_status.clear()
                st.success(f"Updated {affected} filtered row(s) from '{status_choice}' to '{target_status_filter}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Bulk (filter) update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Single update
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Single update")

sel_pk_val: Any = st.selectbox("Pick registration (by PK)", options=pk_options, format_func=lambda v: str(v))
cur_status = df.loc[df[pk] == sel_pk_val, STATUS_FIELD].iloc[0] if (df[pk] == sel_pk_val).any() else None
st.metric("Current status", str(cur_status) if cur_status is not None else "—")

try:
    default_idx = ALLOWED_STATUSES.index(str(cur_status)) if cur_status in ALLOWED_STATUSES else 0
except Exception:
    default_idx = 0

new_status_single = st.selectbox("Set status to", ALLOWED_STATUSES, index=default_idx, key="single_set")
if st.button("Update this row only"):
    try:
        affected = write_update_single(schema, pk, sel_pk_val, new_status_single)
        load_preview.clear(); count_by_status.clear()
        st.success(f"Row {sel_pk_val} updated to '{new_status_single}' (affected={affected}).")
        st.rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")
