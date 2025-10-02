# pages/db_crud.py
import streamlit as st
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from db_handler import DatabaseManager

st.set_page_config(page_title="DB Data Editor (CRUD)", layout="wide")
st.title("🛠️ DB Data Editor (CRUD)")

db = DatabaseManager()

# ───────────────────────────────────────────────────────────────
# Helpers: metadata + lightweight DML execution
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=60)
def load_base_tables() -> pd.DataFrame:
    q = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name
    """
    df = db.fetch_data(q)
    if df is None or df.empty:
        return pd.DataFrame(columns=["table_schema", "table_name"])
    for col in ["table_schema", "table_name"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[["table_schema", "table_name"]]

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
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "ordinal_position", "column_name", "data_type", "is_nullable", "column_default"
        ])
    for col in ["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[["ordinal_position", "column_name", "data_type", "is_nullable", "column_default"]]

@st.cache_data(show_spinner=False, ttl=60)
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
    if df is None or df.empty or "column_name" not in df.columns:
        return []
    return [c for c in df["column_name"].tolist() if c]

def execute_sql(sql: str, params: Optional[tuple] = None):
    """
    Try to run DML via DatabaseManager. Prefer an explicit execute() method
    if present; otherwise fall back to fetch_data (some adapters allow it).
    """
    if hasattr(db, "execute"):
        return getattr(db, "execute")(sql, params)  # type: ignore[attr-defined]
    else:
        return db.fetch_data(sql, params)

def fetch_preview(schema: str, table: str, limit: int = 50) -> pd.DataFrame:
    q = f'SELECT * FROM "{schema}"."{table}" LIMIT {int(limit)}'
    df = db.fetch_data(q)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def parse_value(raw: str, dtype: str):
    """
    Simple parser: blank -> None (NULL).
    Otherwise leave as str for DB to cast, except booleans + json.
    Keep it uncomplicated on purpose.
    """
    if raw is None or raw == "":
        return None
    t = (dtype or "").lower()
    if t == "boolean":
        truthy = {"true", "t", "1", "yes", "y"}
        falsy = {"false", "f", "0", "no", "n"}
        s = raw.strip().lower()
        if s in truthy:
            return True
        if s in falsy:
            return False
        # if not recognized, pass raw and let DB try to cast
        return raw
    if "json" in t:
        try:
            return json.loads(raw)
        except Exception:
            return raw  # let DB validate/complain if needed
    # For everything else, just pass the string—DB will cast if compatible
    return raw

def is_generated(col_default: Any) -> bool:
    """Detect auto/identity columns (e.g., nextval(...) or identity)."""
    if not col_default or not isinstance(col_default, str):
        return False
    s = col_default.lower()
    return "nextval(" in s or "identity" in s or "uuid_generate_v" in s

# ───────────────────────────────────────────────────────────────
# Pick a table
# ───────────────────────────────────────────────────────────────
tables_df = load_base_tables()

if tables_df.empty:
    st.info("No base tables found (or insufficient privileges).")
    st.stop()

schemas = sorted(tables_df["table_schema"].unique().tolist())
c1, c2, c3 = st.columns([1, 1.2, 1])
with c1:
    schema = st.selectbox("Schema", options=schemas)
with c2:
    tnames = sorted(tables_df.loc[tables_df["table_schema"] == schema, "table_name"].tolist())
    table = st.selectbox("Table", options=tnames)
with c3:
    if st.button("🔄 Refresh metadata", use_container_width=True):
        load_base_tables.clear(); get_columns.clear(); get_primary_key.clear()
        st.rerun()

cols_df = get_columns(schema, table)
pk_cols = get_primary_key(schema, table)

with st.expander("Table info", expanded=False):
    left, right = st.columns([2, 1])
    with left:
        st.write("**Columns**")
        st.dataframe(cols_df, use_container_width=True, hide_index=True)
    with right:
        st.write("**Primary key**")
        if pk_cols:
            st.code(", ".join(pk_cols))
        else:
            st.info("No primary key detected.")

st.subheader("Quick preview")
pv_c1, pv_c2 = st.columns([1, 6])
with pv_c1:
    pv_limit = st.number_input("Rows", min_value=1, max_value=1000, value=25, step=5)
with pv_c2:
    st.dataframe(fetch_preview(schema, table, pv_limit), use_container_width=True)

st.divider()

# ───────────────────────────────────────────────────────────────
# CRUD Tabs
# ───────────────────────────────────────────────────────────────
t_ins, t_upd, t_del = st.tabs(["➕ Insert", "✏️ Update", "🗑️ Delete"])

# ───────────────────────────────────────────────────────────────
# INSERT
# ───────────────────────────────────────────────────────────────
with t_ins:
    st.caption("Leave a field **blank** to insert NULL or use the column’s default.")
    ins_cols = cols_df.copy()
    # Suggest hiding generated columns
    hide_auto = st.checkbox("Hide auto/identity columns", value=True)
    if hide_auto:
        ins_cols = ins_cols.loc[~ins_cols["column_default"].apply(is_generated)]

    with st.form("ins_form", clear_on_submit=False):
        values: Dict[str, Any] = {}
        for _, r in ins_cols.iterrows():
            col = r["column_name"]
            dtype = r["data_type"]
            ph = f'{col} ({dtype})'
            raw = st.text_input(ph, key=f"ins_{col}")
            values[col] = parse_value(raw, dtype)

        submitted = st.form_submit_button("Insert row", use_container_width=True)
        if submitted:
            # Only include columns the user actually provided (non-None)
            set_cols = [c for c, v in values.items() if v is not None]
            params = tuple(values[c] for c in set_cols)
            if not set_cols:
                st.warning("Provide at least one value (or unhide auto columns).")
            else:
                cols_sql = ", ".join(f'"{c}"' for c in set_cols)
                ph_sql = ", ".join(["%s"] * len(set_cols))
                sql = f'INSERT INTO "{schema}"."{table}" ({cols_sql}) VALUES ({ph_sql})'
                try:
                    execute_sql(sql, params)
                    st.success("✅ Inserted row.")
                except Exception as e:
                    st.error(f"Insert failed: {e}")

# ───────────────────────────────────────────────────────────────
# UPDATE
# ───────────────────────────────────────────────────────────────
with t_upd:
    if not pk_cols:
        st.info("This table has no primary key; update UI is disabled to avoid accidental mass updates.")
    else:
        st.caption("Provide **primary key** values to locate a row, then edit non-PK columns.")

        # Step 1: locate row by PK
        with st.form("load_row_form"):
            pk_values = []
            for pk in pk_cols:
                dtype = cols_df.loc[cols_df["column_name"] == pk, "data_type"].iat[0]
                val = st.text_input(f'{pk} (PK, {dtype})', key=f"upd_pk_{pk}")
                pk_values.append(parse_value(val, dtype))
            find_btn = st.form_submit_button("Load row", use_container_width=True)

        row_key = f"row_{schema}_{table}_{'_'.join(pk_cols)}"
        if find_btn:
            try:
                where = " AND ".join([f'"{pk}" = %s' for pk in pk_cols])
                sql = f'SELECT * FROM "{schema}"."{table}" WHERE {where} LIMIT 1'
                df = db.fetch_data(sql, tuple(pk_values))
                if df is None or df.empty:
                    st.warning("No row found for the given primary key.")
                    st.session_state.pop(row_key, None)
                else:
                    st.session_state[row_key] = df.iloc[0].to_dict()
                    st.success("Row loaded. Scroll down to edit.")
            except Exception as e:
                st.error(f"Load failed: {e}")
                st.session_state.pop(row_key, None)

        # Step 2: edit and save
        if row_key in st.session_state:
            current: Dict[str, Any] = st.session_state[row_key]
            with st.form("edit_row_form"):
                st.write("**Edit values** (blank → set NULL)")
                new_vals: Dict[str, Any] = {}
                for _, r in cols_df.iterrows():
                    col = r["column_name"]
                    dtype = r["data_type"]
                    if col in pk_cols:
                        st.text_input(f'{col} (PK, {dtype})', value=str(current.get(col)), disabled=True, key=f"edit_{col}")
                        continue
                    raw_default = "" if current.get(col) is None else str(current.get(col))
                    raw = st.text_input(f'{col} ({dtype})', value=raw_default, key=f"edit_{col}")
                    new_vals[col] = parse_value(raw, dtype)

                save_btn = st.form_submit_button("Save changes", use_container_width=True)

            if save_btn:
                try:
                    set_cols = [c for c in new_vals.keys()]
                    set_sql = ", ".join([f'"{c}" = %s' for c in set_cols])
                    where = " AND ".join([f'"{pk}" = %s' for pk in pk_cols])
                    params = tuple(new_vals[c] for c in set_cols) + tuple(current[pk] for pk in pk_cols)
                    sql = f'UPDATE "{schema}"."{table}" SET {set_sql} WHERE {where}'
                    execute_sql(sql, params)
                    st.success("✅ Updated row.")
                    # Refresh the loaded row so defaults/computed cols reflect latest
                    try:
                        df = db.fetch_data(
                            f'SELECT * FROM "{schema}"."{table}" WHERE {where} LIMIT 1',
                            tuple(current[pk] for pk in pk_cols)
                        )
                        if df is not None and not df.empty:
                            st.session_state[row_key] = df.iloc[0].to_dict()
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Update failed: {e}")

# ───────────────────────────────────────────────────────────────
# DELETE
# ───────────────────────────────────────────────────────────────
with t_del:
    if not pk_cols:
        st.info("This table has no primary key; delete UI is disabled to avoid accidental mass deletes.")
    else:
        st.caption("Provide **primary key** values to delete a single row.")
        with st.form("delete_form"):
            del_pk_values = []
            for pk in pk_cols:
                dtype = cols_df.loc[cols_df["column_name"] == pk, "data_type"].iat[0]
                val = st.text_input(f'{pk} (PK, {dtype})', key=f"del_pk_{pk}")
                del_pk_values.append(parse_value(val, dtype))
            confirm = st.checkbox("I understand this will permanently delete the row.")
            delete_btn = st.form_submit_button("Delete row", use_container_width=True)

        if delete_btn:
            if not confirm:
                st.warning("Please confirm before deleting.")
            else:
                try:
                    where = " AND ".join([f'"{pk}" = %s' for pk in pk_cols])
                    sql = f'DELETE FROM "{schema}"."{table}" WHERE {where}'
                    execute_sql(sql, tuple(del_pk_values))
                    st.success("🗑️ Deleted row (if it existed).")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
