# pages/1_Create_Database.py
import re
import pandas as pd
import streamlit as st

# ← change this to your actual file/module name
from db_handler import DatabaseManager, get_conn

st.title("Create Database")

# ----------------------------
# Helpers
# ----------------------------
_DB_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")

def valid_db(name: str) -> bool:
    return bool(_DB_NAME_RE.match(name or ""))

def list_databases(admin_conn) -> list[str]:
    """
    Return a list of database names that allow connections and are not templates.
    """
    sql = """
        SELECT datname
        FROM pg_database
        WHERE datistemplate = FALSE
          AND datallowconn  = TRUE
        ORDER BY datname;
    """
    cur = admin_conn.cursor()
    try:
        cur.execute(sql)
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()

def make_admin_conn(dm: DatabaseManager):
    """
    Build a one-off connection to the 'postgres' database using your handler's
    settings (required for CREATE DATABASE). Ensures autocommit for DDL.
    """
    admin_cfg = dict(dm.cfg)
    admin_cfg["db"] = "postgres"
    # Reuse the same cached-session key to keep one connector per user
    conn = get_conn(admin_cfg, dm._key)  # uses your @st.cache_resource
    # CREATE DATABASE must be executed in autocommit
    try:
        conn.autocommit = True  # pg8000 supports .autocommit
    except Exception:
        pass
    return conn

def run_sql_in_new_db(dm: DatabaseManager, db_name: str, sql_text: str):
    """
    Execute arbitrary SQL inside the newly created database.
    If the SQL returns rows, display them; otherwise commit.
    """
    run_cfg = dict(dm.cfg)
    run_cfg["db"] = db_name
    conn = get_conn(run_cfg, dm._key)
    cur = conn.cursor()
    try:
        # Per-query timeout safeguard (optional)
        cur.execute("SET LOCAL statement_timeout = 8000;")
        cur.execute(sql_text)
        if cur.description:
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
        else:
            conn.commit()
    finally:
        cur.close()
    st.success("Extra SQL executed.")

# ----------------------------
# UI
# ----------------------------
db_name = st.text_input(
    "Database name",
    max_chars=32,
    help="Letters, numbers, underscores; must start with a letter (max 32 chars).",
)
sql_extra = st.text_area(
    "Optional SQL to run in the new database (e.g. CREATE TABLE …)",
    height=180,
    placeholder="-- Paste your DDL here (tables, views, etc.)"
)

col1, col2 = st.columns(2)
with col1:
    create_clicked = st.button("Create Database and Run SQL", type="primary")
with col2:
    list_clicked = st.button("List existing databases")

# Instantiate your DB manager once; pulls config from env/streamlit secrets
dm = DatabaseManager()

if create_clicked:
    if not valid_db(db_name):
        st.error("Invalid database name. Use letters, numbers, underscores; start with a letter (max 32).")
    else:
        try:
            admin_conn = make_admin_conn(dm)
            # CREATE DATABASE cannot be parameterized as an identifier; we validated above and quote it
            qname = f'"{db_name}"'
            with admin_conn.cursor() as cur:
                # Optional: set a short statement timeout to keep UI snappy
                try:
                    cur.execute("SET LOCAL statement_timeout = 8000;")
                except Exception:
                    pass
                cur.execute(f"CREATE DATABASE {qname};")
            st.success(f"Database **{db_name}** created.")

            if sql_extra.strip():
                run_sql_in_new_db(dm, db_name, sql_extra)

        except Exception as e:
            # Handle "database already exists" cleanly (PostgreSQL error code 42P04)
            sqlstate = getattr(e, "sqlstate", None) or getattr(e, "pgcode", None)
            msg = str(e)
            if sqlstate == "42P04" or "already exists" in msg.lower():
                st.warning("Database already exists.")
            else:
                st.error(msg)

if list_clicked:
    try:
        admin_conn = make_admin_conn(dm)
        st.dataframe(
            pd.DataFrame({"Database": list_databases(admin_conn)}),
            use_container_width=True
        )
    except Exception as e:
        st.error(str(e))
