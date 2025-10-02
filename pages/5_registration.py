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
# Helpers (SELECT-only via db.fetch_data)
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
    return df["table_schema"].iat[0] if isinstance(df, pd.DataFrame) and not df.empty else "public"

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
    return df["column_name"].tolist() if isinstance(df, pd.DataFrame) and not df.empty else []

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

def update_single(schema: str, pk_col: str, pk_val, new_status: str) -> int:
    """
    Update one row but always return a tiny result set so fetch_data is satisfied.
    """
    sql = f'''
        WITH upd AS (
          UPDATE "{schema}"."{TABLE}"
             SET "{STATUS_FIELD}" = %s
           WHERE "{pk_col}" = %s
         RETURNING 1
        )
        SELECT COUNT(*)::int AS affected FROM upd;
    '''
    df = db.fetch_data(sql, (new_status, pk_val))
    if isinstance(df, pd.DataFrame) and not df.empty and "affected" in df.columns:
        return int(df["affected"].iat[0])
    return 0

def update_bulk_ids(schema: str, pk_col: str, id_list: List, new_status: str) -> int:
    """
    Bulk update selected ids using ANY(array) and return affected count.
    """
    if not id_list:
        return 0
    sql = f'''
        WITH upd AS (
          UPDATE "{schema}"."{TABLE}" r
             SET "{STATUS_FIELD}" = %s
           WHERE r."{pk_col}" = ANY(%s)
         RETURNING 1
        )
        SELECT COUNT(*)::int AS affected FROM upd;
    '''
    # psycopg2 will adapt Python list -> SQL array
    df = db.fetch_data(sql, (new_status, id_list))
    if isinstance(df, pd.DataFrame) and not df.empty and "affected" in df.columns:
        return int(df["affected"].iat[0])
    return 0

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
# Bulk update (selected IDs)
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bulk update enrollment status")

ids = st.multiselect("Select registrations by primary key", df[pk].astype(str).tolist())
# Keep the actual types for DB by mapping back (stringify for UI safety)
ids_typed = [df.loc[df[pk].astype(str) == s, pk].iloc[0] for s in ids] if ids else []

bulk_status = st.selectbox("New status for selected", ALLOWED_STATUSES, index=0)
if st.button("✅ Apply to selected", type="primary"):
    try:
        affected = update_bulk_ids(schema, pk, ids_typed, bulk_status)
        load_preview.clear()
        st.success(f"Updated {affected} row(s) to '{bulk_status}'.")
        st.rerun()
    except Exception as e:
        st.error(f"Bulk update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Bulk apply to all in current preview
# ───────────────────────────────────────────────────────────────
st.caption("Or apply to *all rows currently shown* in the preview above.")
preview_status = st.selectbox("New status for all in preview", ALLOWED_STATUSES, index=0, key="preview_status")
if st.button("⚡ Apply to all in preview"):
    try:
        preview_ids = df[pk].tolist()
        affected = update_bulk_ids(schema, pk, preview_ids, preview_status)
        load_preview.clear()
        st.success(f"Updated {affected} previewed row(s) to '{preview_status}'.")
        st.rerun()
    except Exception as e:
        st.error(f"Bulk (preview) update failed: {e}")

# ───────────────────────────────────────────────────────────────
# Single-row update
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Single update (by primary key)")

sel_pk_str = st.selectbox("Pick registration", df[pk].astype(str).tolist())
sel_pk_val = df.loc[df[pk].astype(str) == sel_pk_str, pk].iloc[0]
cur_status = df.loc[df[pk].astype(str) == sel_pk_str, STATUS_FIELD].iloc[0]

st.metric("Current status", str(cur_status))
single_status = st.selectbox("Set status to", ALLOWED_STATUSES, index=ALLOWED_STATUSES.index(str(cur_status)))

if st.button("Update this row only"):
    try:
        affected = update_single(schema, pk, sel_pk_val, single_status)
        load_preview.clear()
        st.success(f"Row {sel_pk_str} updated to '{single_status}' (affected={affected}).")
        st.rerun()
    except Exception as e:
        st.error(f"Update failed: {e}")
