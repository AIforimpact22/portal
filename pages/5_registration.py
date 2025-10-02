# pages/registrations_status_sql.py
import streamlit as st
import pandas as pd
from typing import List, Any, Optional, Tuple
from db_handler import DatabaseManager  # READ-ONLY

st.set_page_config(page_title="Registrations — Generate UPDATE SQL", layout="wide")
st.title("📝 Registrations — Generate UPDATE SQL (safe, no DB writes)")

TABLE = "registrations"
STATUS_FIELD = "enrollment_status"
ALLOWED_STATUSES = ["pending", "accepted", "rejected"]

db = DatabaseManager()  # used only for SELECTs

# ---------- helpers (read-only) ----------
@st.cache_data(show_spinner=False, ttl=120)
def find_schema_for_table(table_name: str) -> str:
    q = """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_name = %s
          AND table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY 1 LIMIT 1
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

@st.cache_data(show_spinner=True, ttl=60)
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

def gen_update_ids(schema: str, pk_col: str, ids: List[Any], new_status: str) -> str:
    if not ids:
        return "-- No IDs provided"
    # Make a VALUES list preserving types (strings quoted, numbers as-is)
    def lit(v: Any) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            # ints/floats: use raw (floats are rare for pk, but safe here)
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        # quote text
        s = str(v).replace("'", "''")
        return f"'{s}'"
    values = ", ".join(f"({lit(v)})" for v in ids)
    new = new_status.replace("'", "''")
    return f"""
-- Bulk set selected IDs to '{new_status}'
UPDATE "{schema}"."{TABLE}" AS r
SET "{STATUS_FIELD}" = '{new}'
FROM (VALUES {values}) AS v(id)
WHERE r."{pk_col}" = v.id;
"""

def gen_update_filter(schema: str, from_status: str, to_status: str) -> str:
    f = from_status.replace("'", "''")
    t = to_status.replace("'", "''")
    return f"""
-- Bulk set all rows with status='{from_status}' to '{to_status}'
UPDATE "{schema}"."{TABLE}"
SET "{STATUS_FIELD}" = '{t}'
WHERE "{STATUS_FIELD}" = '{f}';
"""

def gen_update_single(schema: str, pk_col: str, pk_val: Any, new_status: str) -> str:
    def lit(v: Any) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        return f"'{str(v).replace("'", "''")}'"
    new = new_status.replace("'", "''")
    return f"""
-- Set one row
UPDATE "{schema}"."{TABLE}"
SET "{STATUS_FIELD}" = '{new}'
WHERE "{pk_col}" = {lit(pk_val)};
"""

# ---------- UI ----------
schema = find_schema_for_table(TABLE)
pk_cols = get_primary_key(schema, TABLE)
if len(pk_cols) != 1:
    st.error("This page requires a single-column primary key on 'registrations'.")
    if pk_cols:
        st.info(f"Detected PK columns: {', '.join(pk_cols)}")
    st.stop()
pk = pk_cols[0]

st.subheader("Browse")
c1, c2, c3 = st.columns([1.3, 1, 1])
with c1:
    status_choice = st.radio("Filter by status", ["pending", "accepted", "rejected", "all"], index=0, horizontal=True)
with c2:
    limit = st.number_input("Preview rows", 5, 2000, 100, step=25)
with c3:
    if st.button("🔄 Refresh", use_container_width=True):
        find_schema_for_table.clear(); get_primary_key.clear(); load_preview.clear()
        st.rerun()

df = load_preview(schema, status_choice, int(limit), pk)
if df.empty:
    st.info("No rows found for current filter.")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

pk_options: List[Any] = pd.Series(df[pk]).dropna().astype(object).unique().tolist()

st.divider()
st.subheader("Generate SQL")

t1, t2, t3 = st.tabs(["➕ Selected IDs", "⚡ All in preview / filter", "✏️ Single row"])

# --- Tab 1: Selected IDs ---
with t1:
    sel_ids: List[Any] = st.multiselect(
        "Pick registrations by primary key",
        options=pk_options,
        format_func=lambda v: str(v),
    )
    new_status_sel = st.selectbox("Set selected to", ALLOWED_STATUSES, index=0)
    sql1 = gen_update_ids(schema, pk, sel_ids, new_status_sel) if sel_ids else "-- Select at least one ID"
    st.code(sql1.strip(), language="sql")
    st.download_button("⬇️ Download SQL (selected)", sql1.strip().encode("utf-8"), file_name="update_selected.sql", mime="text/plain")

# --- Tab 2: All in preview / filter ---
with t2:
    colA, colB = st.columns(2)
    with colA:
        status_prev = st.selectbox("Set **all in current preview** to", ALLOWED_STATUSES, index=0, key="prev")
        sql2a = gen_update_ids(schema, pk, df[pk].dropna().tolist(), status_prev)
        st.code(sql2a.strip(), language="sql")
        st.download_button("⬇️ Download SQL (preview)", sql2a.strip().encode("utf-8"), file_name="update_preview.sql", mime="text/plain")
    with colB:
        st.caption("Or change by DB filter (doesn’t depend on preview limit).")
        from_status = st.selectbox("Change all rows with status", ["pending", "accepted", "rejected"], index=0, key="from")
        to_status = st.selectbox("…to", ALLOWED_STATUSES, index=1, key="to")
        sql2b = gen_update_filter(schema, from_status, to_status)
        st.code(sql2b.strip(), language="sql")
        st.download_button("⬇️ Download SQL (filter)", sql2b.strip().encode("utf-8"), file_name="update_filter.sql", mime="text/plain")

# --- Tab 3: Single row ---
with t3:
    sel_pk = st.selectbox("Pick one registration", options=pk_options, format_func=lambda v: str(v))
    current = df.loc[df[pk] == sel_pk, STATUS_FIELD]
    st.metric("Current status (preview)", current.iloc[0] if not current.empty else "—")
    new_status_one = st.selectbox("Set status to", ALLOWED_STATUSES, index=0, key="single")
    sql3 = gen_update_single(schema, pk, sel_pk, new_status_one)
    st.code(sql3.strip(), language="sql")
    st.download_button("⬇️ Download SQL (single)", sql3.strip().encode("utf-8"), file_name="update_single.sql", mime="text/plain")

st.info("Copy the generated SQL and run it in psql/pgAdmin/your backend console. No DB writes are executed from this page.")
