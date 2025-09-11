# pages/db_tables.py
import streamlit as st
import pandas as pd
from db_handler import DatabaseManager

st.set_page_config(page_title="DB Tables Browser", layout="wide")
st.title("📚 Database Tables Browser")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Data access (return shaped DataFrames, never column-less)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def load_objects(include_views: bool) -> pd.DataFrame:
    q = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    """
    if not include_views:
        q += " AND table_type = 'BASE TABLE'"
    q += " ORDER BY table_schema, table_name"
    df = db.fetch_data(q)
    # Ensure shaped result
    if df.empty:
        return pd.DataFrame(columns=["table_schema", "table_name", "table_type"])
    # Some drivers can give lowercase/uppercase inconsistencies; normalize:
    for col in ["table_schema", "table_name", "table_type"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[["table_schema", "table_name", "table_type"]]

@st.cache_data(show_spinner=False, ttl=60)
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
    if df.empty:
        return pd.DataFrame(columns=[
            "ordinal_position", "column_name", "data_type", "is_nullable", "column_default"
        ])
    # Normalize expected columns
    for col in ["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]]

@st.cache_data(show_spinner=True, ttl=30)
def get_row_count(schema: str, table: str) -> int:
    df = db.fetch_data(f'SELECT COUNT(*) AS c FROM "{schema}"."{table}"')
    if df.empty or "c" not in df.columns:
        return 0
    return int(df["c"].iat[0])

@st.cache_data(show_spinner=True, ttl=30)
def preview_table(schema: str, table: str, limit: int) -> pd.DataFrame:
    df = db.fetch_data(f'SELECT * FROM "{schema}"."{table}" LIMIT {int(limit)}')
    # If empty with no columns, return shaped empty DF
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df

# ───────────────────────────────────────────────────────────────
# Controls
# ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
with c1:
    include_views = st.toggle("Include views", value=False)
with c2:
    with_counts = st.toggle("Show row counts (slow)", value=False,
                            help="Runs COUNT(*) per table.")
with c3:
    default_limit = st.number_input("Preview rows", 5, 1000, 50, 5)
with c4:
    name_filter = st.text_input("Filter name contains", "", label_visibility="collapsed")

# ───────────────────────────────────────────────────────────────
# Load + filter (safe even if empty)
# ───────────────────────────────────────────────────────────────
objects_df = load_objects(include_views)

if objects_df.empty:
    st.info("No tables/views found (or insufficient privileges).")
    st.stop()

schemas = sorted(objects_df["table_schema"].fillna("").unique().tolist())
if not schemas:
    st.info("No schemas available to display.")
    st.stop()

schema_pick = st.multiselect("Schemas", options=schemas, default=schemas)

filtered = objects_df[objects_df["table_schema"].isin(schema_pick)].copy()
if name_filter:
    nf = name_filter.lower()
    filtered = filtered[filtered["table_name"].str.lower().str.contains(nf, na=False)]

if filtered.empty:
    st.info("No objects match the current filters.")
    st.stop()

# Optional row counts
if with_counts:
    st.info("Counting rows… This may take a moment on large tables.", icon="⏳")
    counts = []
    for _, r in filtered.iterrows():
        try:
            counts.append(get_row_count(r["table_schema"], r["table_name"]))
        except Exception:
            counts.append(None)
    if "row_count" in filtered.columns:
        filtered = filtered.drop(columns=["row_count"])
    filtered.insert(2, "row_count", counts)

st.subheader("Objects")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────
# Inspect a selected table/view
# ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Inspect a table / view")

choices = [f'{s}.{t}' for s, t in zip(filtered["table_schema"], filtered["table_name"])]
selection = st.selectbox("Pick an object", options=choices if choices else ["— none —"])

if not choices:
    st.stop()

schema, table = selection.split(".", 1)

left, right = st.columns([1, 3])
with left:
    if st.button("🔄 Refresh", use_container_width=True):
        load_objects.clear(); get_columns.clear()
        get_row_count.clear(); preview_table.clear()
        st.rerun()

    show_cols = st.checkbox("Show columns", True)
    do_preview = st.checkbox("Preview data", True)
    limit = st.number_input("Limit", 1, 5000, int(default_limit), 10)

    if with_counts:
        try:
            st.metric("Row count", f"{get_row_count(schema, table):,}")
        except Exception as e:
            st.warning(f"Row count failed: {e}")

with right:
    tabs = st.tabs(["Columns", "Preview", "SQL"])

    with tabs[0]:
        if show_cols:
            try:
                cols_df = get_columns(schema, table)
                st.dataframe(cols_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to load columns: {e}")

    with tabs[1]:
        if do_preview:
            try:
                df = preview_table(schema, table, limit)
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{schema}.{table}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Preview failed: {e}")

    with tabs[2]:
        st.code(f'SELECT * FROM "{schema}"."{table}" LIMIT {int(limit)};', language="sql")
