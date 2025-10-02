# pages/registrations_status.py
import streamlit as st
import pandas as pd
from typing import List
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
def find_schema() -> str:
    q = """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_name = %s
          AND table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema')
        LIMIT 1
    """
    df = db.fetch_data(q, (TABLE,))
    if df is None or df.empty:
        return "public"
    return df["table_schema"].iat[0]

@st.cache_data(show_spinner=False, ttl=120)
def get_primary_key(schema: str) -> List[str]:
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
    df = db.fetch_data(q, (schema, TABLE))
    if df is None or df.empty:
        return []
    return df["column_name"].tolist()

def safe_execute(sql: str, params: tuple | None = None):
    """
    Run a DML statement safely with DatabaseManager.
    Uses fetch_data but discards results (since UPDATE/DELETE return nothing).
    """
    try:
        _ = db.fetch_data(sql, params)
    except Exception as e:
        # some drivers complain about no results to fetch -> ignore
        if "no results" in str(e).lower():
            return
        raise

@st.cache_data(show_spinner=True, ttl=30)
def load_preview(schema: str, status: str, limit: int) -> pd.DataFrame:
    sql = f'SELECT * FROM "{schema}"."{TABLE}"'
    params = None
    if status != "all":
        sql += f' WHERE "{STATUS_FIELD}" = %s'
        params = (status,)
    sql += f' ORDER BY 1 LIMIT {int(limit)}'
    df = db.fetch_data(sql, params)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# ───────────────────────────────────────────────────────────────
# Main UI
# ───────────────────────────────────────────────────────────────
schema = find_schema()
pk_cols = get_primary_key(schema)
if len(pk_cols) != 1:
    st.error("This page requires a single-column primary key on 'registrations'.")
    st.stop()
pk = pk_cols[0]

st.subheader("Browse registrations")
c1, c2 = st.columns([1, 2])
with c1:
    status_choice = st.radio("Filter by status", ["pending", "accepted", "rejected", "all"], index=0)
with c2:
    limit = st.number_input("Preview rows", 5, 1000, 100, step=25)

df = load_preview(schema, status_choice, limit)
if df.empty:
    st.info("No rows found.")
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────
# Bulk update
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bulk update enrollment status")

ids = st.multiselect("Select registrations by primary key", df[pk].tolist(), format_func=str)
new_status = st.selectbox("New status", ALLOWED_STATUSES, index=0)
if st.button("✅ Apply to selected", type="primary"):
    if not ids:
        st.warning("Pick at least one registration.")
    else:
        try:
            placeholders = ",".join(["%s"] * len(ids))
            sql = f'UPDATE "{schema}"."{TABLE}" SET "{STATUS_FIELD}" = %s WHERE "{pk}" IN ({placeholders})'
            params = tuple([new_status] + ids)
            safe_execute(sql, params)
            load_preview.clear()
            st.success(f"Updated {len(ids)} row(s) to '{new_status}'.")
            st.rerun()
        except Exception as e:
            st.error(f"Bulk update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Single update
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Single update (by primary key)")

sel_pk = st.selectbox("Pick registration", df[pk].tolist(), format_func=str)
cur_status = df.loc[df[pk] == sel_pk, STATUS_FIELD].iat[0]
st.metric("Current status", cur_status)

new_status_single = st.selectbox("Set status to", ALLOWED_STATUSES, index=ALLOWED_STATUSES.index(cur_status))
if st.button("Update this row only"):
    try:
        sql = f'UPDATE "{schema}"."{TABLE}" SET "{STATUS_FIELD}" = %s WHERE "{pk}" = %s'
        safe_execute(sql, (new_status_single, sel_pk))
        load_preview.clear()
        st.success(f"Row {sel_pk} updated to '{new_status_single}'.")
        st.rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")
