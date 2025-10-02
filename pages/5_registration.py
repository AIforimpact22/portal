# pages/registrations_status.py
import streamlit as st
import pandas as pd
from typing import List, Optional
from db_handler import DatabaseManager

st.set_page_config(page_title="Registrations — Enrollment Status", layout="wide")
st.title("📝 Registrations — Enrollment Status")

db = DatabaseManager()

TABLE = "registrations"
STATUS_FIELD = "enrollment_status"
ALLOWED_STATUSES = ["pending", "accepted", "rejected"]

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def find_schemas_for_table(table_name: str) -> List[str]:
    q = """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_name = %s
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema
    """
    df = db.fetch_data(q, (table_name,))
    if df is None or df.empty:
        return []
    return df["table_schema"].tolist()

@st.cache_data(show_spinner=False, ttl=120)
def get_primary_key(schema: str, table: str) -> List[str]:
    q = """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
          AND tc.table_name = %s
        ORDER BY kcu.ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df is None or df.empty or "column_name" not in df.columns:
        return []
    return [c for c in df["column_name"].tolist() if c]

@st.cache_data(show_spinner=False, ttl=120)
def get_columns(schema: str, table: str) -> pd.DataFrame:
    q = """
        SELECT
            ordinal_position,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    df = db.fetch_data(q, (schema, table))
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ordinal_position", "column_name", "data_type", "is_nullable", "column_default"
        ])
    for col in ["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]]

def execute_sql(sql: str, params: Optional[tuple] = None):
    """Prefer an explicit execute() if your DatabaseManager exposes one."""
    if hasattr(db, "execute"):
        return getattr(db, "execute")(sql, params)  # type: ignore
    else:
        return db.fetch_data(sql, params)

@st.cache_data(show_spinner=False, ttl=30)
def count_by_status(schema: str) -> pd.DataFrame:
    q = f'''
        SELECT "{STATUS_FIELD}"::text AS status, COUNT(*)::bigint AS count
        FROM "{schema}"."{TABLE}"
        GROUP BY 1
        ORDER BY 1
    '''
    df = db.fetch_data(q)
    if df is None or df.empty:
        return pd.DataFrame({"status": [], "count": []})
    return df

@st.cache_data(show_spinner=True, ttl=30)
def load_preview(schema: str, status_choice: str, limit: int, order_by: Optional[str]) -> pd.DataFrame:
    sql = f'SELECT * FROM "{schema}"."{TABLE}"'
    params: tuple | None = None
    if status_choice != "all":
        sql += f' WHERE "{STATUS_FIELD}" = %s'
        params = (status_choice,)
    if order_by:
        sql += f' ORDER BY "{order_by}"'
    sql += f' LIMIT {int(limit)}'
    df = db.fetch_data(sql, params)
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df

def get_status_count(df: pd.DataFrame, name: str) -> int:
    if df is None or df.empty:
        return 0
    s = df.loc[df["status"] == name, "count"]
    return int(s.iat[0]) if not s.empty else 0

# ───────────────────────────────────────────────────────────────
# Schema + metadata
# ───────────────────────────────────────────────────────────────
schemas_found = find_schemas_for_table(TABLE)
schema = None
c_schema, c_refresh = st.columns([2, 1])
with c_schema:
    if schemas_found:
        schema = st.selectbox("Schema", options=schemas_found, index=0)
    else:
        schema = st.text_input("Schema", value="public", help="Schema where the 'registrations' table lives.")
with c_refresh:
    if st.button("🔄 Refresh", use_container_width=True):
        find_schemas_for_table.clear()
        get_primary_key.clear()
        get_columns.clear()
        count_by_status.clear()
        load_preview.clear()
        st.rerun()

cols_df = get_columns(schema, TABLE)
if STATUS_FIELD not in cols_df["column_name"].values:
    st.error(f"Column '{STATUS_FIELD}' not found in {schema}.{TABLE}.")
    st.stop()

pk_cols = get_primary_key(schema, TABLE)
if len(pk_cols) != 1:
    st.error("This page expects a single-column primary key on 'registrations'.")
    if pk_cols:
        st.info(f"Detected primary key columns: {', '.join(pk_cols)}")
    st.stop()

pk = pk_cols[0]

# ───────────────────────────────────────────────────────────────
# Overview
# ───────────────────────────────────────────────────────────────
st.subheader("Overview")
counts = count_by_status(schema)
m1, m2, m3 = st.columns(3)
m1.metric("pending", f"{get_status_count(counts, 'pending'):,}")
m2.metric("accepted", f"{get_status_count(counts, 'accepted'):,}")
m3.metric("rejected", f"{get_status_count(counts, 'rejected'):,}")

st.divider()

# ───────────────────────────────────────────────────────────────
# Filter + preview
# ───────────────────────────────────────────────────────────────
st.subheader("Browse & pick a registration")

fc1, fc2, fc3 = st.columns([1.2, 1, 2])
with fc1:
    status_choice = st.radio(
        "Filter by current status",
        options=["pending", "accepted", "rejected", "all"],
        index=0,
        horizontal=True,
    )
with fc2:
    limit = st.number_input("Rows to preview", min_value=5, max_value=2000, value=100, step=25)
with fc3:
    st.caption(f"Table: `{schema}.{TABLE}` — Primary key: `{pk}`")

df = load_preview(schema, status_choice, int(limit), order_by=pk)

if df.empty:
    st.info("No rows found for the selected filter/limit.")
    st.stop()

# Display preview + CSV download
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download preview as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{schema}.{TABLE}.{status_choice}.csv",
    mime="text/csv",
)

# ───────────────────────────────────────────────────────────────
# Select a row and update its status
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Update enrollment status")

options = df[pk].dropna().tolist()
if not options:
    st.info("No selectable primary key values in the preview.")
    st.stop()

sel_pk = st.selectbox("Pick a registration (by primary key)", options=options, format_func=lambda x: str(x))

# Fetch current status for the selected row
row = db.fetch_data(
    f'SELECT "{pk}", "{STATUS_FIELD}" FROM "{schema}"."{TABLE}" WHERE "{pk}" = %s LIMIT 1',
    (sel_pk,),
)

if row is None or row.empty:
    st.warning("Could not load the selected row. It may have been modified or deleted.")
    st.stop()

current_status = str(row[STATUS_FIELD].iat[0]) if STATUS_FIELD in row.columns else ""

col_left, col_right = st.columns([1, 2])
with col_left:
    st.metric("Current status", current_status)

with col_right:
    try:
        default_idx = ALLOWED_STATUSES.index(current_status) if current_status in ALLOWED_STATUSES else 0
    except Exception:
        default_idx = 0
    new_status = st.selectbox(
        "Set status to",
        options=ALLOWED_STATUSES,
        index=default_idx,
        help="Choose the new enrollment status and click Update.",
    )
    update = st.button("✅ Update status", type="primary")

if update:
    try:
        sql = f'UPDATE "{schema}"."{TABLE}" SET "{STATUS_FIELD}" = %s WHERE "{pk}" = %s'
        execute_sql(sql, (new_status, sel_pk))
        # Clear caches and refresh the page so counts/preview reflect the change
        count_by_status.clear(); load_preview.clear()
        st.success("Enrollment status updated.")
        st.rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Tips
# ───────────────────────────────────────────────────────────────
with st.expander("Notes", expanded=False):
    st.write(
        """
        - This page only updates the **enrollment_status** column on **registrations**.
        - It requires a **single-column primary key** on the table to safely target a row.
        - If your `DatabaseManager` exposes an `execute(sql, params)` method, it will be used for DML.
          Otherwise it falls back to `fetch_data(sql, params)`.
        """
    )
