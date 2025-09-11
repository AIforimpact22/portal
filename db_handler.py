import os
import uuid
import pandas as pd
import streamlit as st

# Cloud SQL Python Connector (PostgreSQL via pg8000)
from google.cloud.sql.connector import Connector
from google.oauth2 import service_account  # ← explicit creds for off-GCP
import pg8000  # ensure driver is installed


# ───────────────────────────────────────────────────────────────
# 1) One cached Connector + connection per user session
# ───────────────────────────────────────────────────────────────
def _session_key() -> str:
    """Return a unique key for the current user session."""
    if "_session_key" not in st.session_state:
        st.session_state["_session_key"] = uuid.uuid4().hex
    return st.session_state["_session_key"]


@st.cache_resource(show_spinner=False)
def get_conn(cfg: dict, key: str):
    """
    Create (once per session) and return a PostgreSQL connection using
    the Cloud SQL Python Connector with the pg8000 driver.

    Expected cfg keys:
      - instance_connection_name: "PROJECT:REGION:INSTANCE"
      - user: DB user (e.g. "postgres")
      - password: raw DB password (NO URL encoding)
      - db: database name
    Also expects either:
      - st.secrets["gcp_service_account"] block with a service account JSON
        OR
      - GOOGLE_APPLICATION_CREDENTIALS env var pointing to a key file
    """
    # Load explicit credentials if provided in Streamlit secrets
    creds = None
    if "gcp_service_account" in st.secrets:
        creds = service_account.Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"])
        )

    # If creds is None and ADC is configured via env, Connector will pick it up.
    connector = Connector(credentials=creds) if creds else Connector()

    def _connect():
        conn = connector.connect(
            cfg["instance_connection_name"],
            "pg8000",
            user=cfg["user"],
            password=cfg["password"],
            db=cfg["db"],
            timeout=10,            # connect timeout (seconds)
            enable_iam_auth=False, # using DB password auth (not IAM DB Auth)
        )
        # Per-session statement timeout (5s) so no query can hang the UI
        cur = conn.cursor()
        try:
            cur.execute("SET statement_timeout = 5000;")
        finally:
            cur.close()
        return conn

    conn = _connect()

    # Clean up both connection and connector when the Streamlit session ends
    try:
        def _cleanup():
            try:
                conn.close()
            except Exception:
                pass
            try:
                connector.close()
            except Exception:
                pass
        st.on_session_end(_cleanup)
    except Exception:
        pass

    conn._cloudsql_connector = connector  # optional handle
    return conn


# ───────────────────────────────────────────────────────────────
# 2) Database manager with auto-reconnect logic
# ───────────────────────────────────────────────────────────────
class DatabaseManager:
    """General DB interactions using a cached connection (Cloud SQL Connector)."""

    def __init__(self):
        # Prefer env vars (Cloud Run/App Engine); fallback to Streamlit secrets.
        cfg = {
            "instance_connection_name": os.getenv(
                "INSTANCE_CONNECTION_NAME",
                st.secrets["cloudsql"]["instance_connection_name"],
            ),
            "user": os.getenv("DB_USER", st.secrets["cloudsql"]["user"]),
            "password": os.getenv("DB_PASSWORD", st.secrets["cloudsql"]["password"]),
            "db": os.getenv("DB_NAME", st.secrets["cloudsql"]["db"]),
        }
        self.cfg = cfg
        self._key = _session_key()
        self.conn = get_conn(self.cfg, self._key)  # cached per user session

    # ────────── internal helpers ──────────
    def _reconnect(self):
        """Force a reconnection by clearing the cached resource and re-calling it."""
        try:
            get_conn.clear()
        except Exception:
            pass
        self.conn = get_conn(self.cfg, self._key)

    def _ensure_live_conn(self):
        """
        Ensure we have a live connection. Quick ping; if it fails, reconnect.
        """
        try:
            cur = self.conn.cursor()
            try:
                # Make the ping return fast even if server is busy
                cur.execute("SET LOCAL statement_timeout = 2000;")
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
            finally:
                cur.close()
        except Exception:
            # Stale/broken connection → rebuild
            self._reconnect()

    def _fetch_df(self, query: str, params=None) -> pd.DataFrame:
        self._ensure_live_conn()
        try:
            cur = self.conn.cursor()
            try:
                # Per-query timeout safeguard (8s)
                cur.execute("SET LOCAL statement_timeout = 8000;")
                cur.execute(query, params or ())
                rows = cur.fetchall()
                cols = [c[0] for c in cur.description] if cur.description else []
            finally:
                cur.close()
            return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
        except Exception:
            # On any driver/connector hiccup → one reconnect + retry once
            self._reconnect()
            cur = self.conn.cursor()
            try:
                cur.execute("SET LOCAL statement_timeout = 8000;")
                cur.execute(query, params or ())
                rows = cur.fetchall()
                cols = [c[0] for c in cur.description] if cur.description else []
            finally:
                cur.close()
            return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()

    def _execute(self, query: str, params=None, returning=False):
        self._ensure_live_conn()
        try:
            cur = self.conn.cursor()
            try:
                cur.execute("SET LOCAL statement_timeout = 8000;")
                cur.execute(query, params or ())
                res = cur.fetchone() if returning else None
            finally:
                cur.close()
            self.conn.commit()
            return res
        except Exception:
            # Reconnect and retry once
            self._reconnect()
            cur = self.conn.cursor()
            try:
                cur.execute("SET LOCAL statement_timeout = 8000;")
                cur.execute(query, params or ())
                res = cur.fetchone() if returning else None
            finally:
                cur.close()
            self.conn.commit()
            return res

    # ────────── public API ──────────
    def fetch_data(self, query, params=None):
        return self._fetch_df(query, params)

    def execute_command(self, query, params=None):
        self._execute(query, params)

    def execute_command_returning(self, query, params=None):
        return self._execute(query, params, returning=True)

    # ─────────── Dropdown Management ───────────
    def get_all_sections(self):
        df = self.fetch_data("SELECT DISTINCT section FROM dropdowns")
        return df["section"].tolist()

    def get_dropdown_values(self, section):
        q = "SELECT value FROM dropdowns WHERE section = %s"
        df = self.fetch_data(q, (section,))
        return df["value"].tolist()

    # ─────────── Supplier Management ───────────
    def get_suppliers(self):
        return self.fetch_data(
            "SELECT supplierid, suppliername FROM supplier"
        )

    # ─────────── Inventory Management ───────────
    def add_inventory(self, data: dict):
        cols = ", ".join(data.keys())
        ph   = ", ".join(["%s"] * len(data))
        q = f"INSERT INTO inventory ({cols}) VALUES ({ph})"
        self.execute_command(q, list(data.values()))

    # ─────────── Foreign-key checks ───────────
    def check_foreign_key_references(self, referenced_table: str, referenced_column: str, value) -> list[str]:
        """
        Return a list of tables that still reference the given value
        through a FOREIGN KEY constraint. Empty list → safe to delete.
        """
        fk_sql = """
            SELECT tc.table_schema,
                   tc.table_name
            FROM   information_schema.table_constraints AS tc
            JOIN   information_schema.key_column_usage AS kcu
                   ON tc.constraint_name = kcu.constraint_name
            JOIN   information_schema.constraint_column_usage AS ccu
                   ON ccu.constraint_name = tc.constraint_name
            WHERE  tc.constraint_type = 'FOREIGN KEY'
              AND  ccu.table_name      = %s
              AND  ccu.column_name     = %s;
        """
        fks = self.fetch_data(fk_sql, (referenced_table, referenced_column))

        conflicts: list[str] = []
        for _, row in fks.iterrows():
            schema = row["table_schema"]
            table  = row["table_name"]

            exists_sql = f"""
                SELECT EXISTS(
                    SELECT 1
                    FROM   {schema}.{table}
                    WHERE  {referenced_column} = %s
                );
            """
            exists = self.fetch_data(exists_sql, (value,)).iat[0, 0]
            if exists:
                conflicts.append(f"{schema}.{table}")

        return sorted(set(conflicts))
