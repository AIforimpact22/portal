import os
import json
import uuid
import pandas as pd
import streamlit as st

# Cloud SQL Python Connector (PostgreSQL via pg8000)
from google.cloud.sql.connector import Connector, IPTypes
from google.oauth2 import service_account  # explicit creds for off-GCP
import pg8000  # ensure driver is installed


# ───────────────────────────────────────────────────────────────
# Helpers: session key & credentials
# ───────────────────────────────────────────────────────────────
def _session_key() -> str:
    """Return a unique key for the current user session."""
    if "_session_key" not in st.session_state:
        st.session_state["_session_key"] = uuid.uuid4().hex
    return st.session_state["_session_key"]


def _load_explicit_credentials():
    """
    Build service-account Credentials from one of:
      1) st.secrets["gcp_service_account"] (dict)  ← Streamlit preferred
      2) st.secrets["GCP_SA_KEY_JSON"] (string with JSON)
      3) env var GCP_SA_KEY_JSON (string with JSON)
      4) env var GOOGLE_APPLICATION_CREDENTIALS (file path)
    Returns a google.oauth2.service_account.Credentials or raises ValueError.
    """
    # 1) Streamlit secret as a dict-like block
    if "gcp_service_account" in st.secrets:
        info = dict(st.secrets["gcp_service_account"])
        return service_account.Credentials.from_service_account_info(info)

    # 2) Streamlit secret JSON string
    if "GCP_SA_KEY_JSON" in st.secrets:
        info = json.loads(st.secrets["GCP_SA_KEY_JSON"])
        return service_account.Credentials.from_service_account_info(info)

    # 3) Env var JSON string
    if os.getenv("GCP_SA_KEY_JSON"):
        info = json.loads(os.environ["GCP_SA_KEY_JSON"])
        return service_account.Credentials.from_service_account_info(info)

    # 4) Env var file path
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        )

    raise ValueError(
        "No service account credentials found. Provide one of: "
        "st.secrets['gcp_service_account'], st.secrets['GCP_SA_KEY_JSON'], "
        "env GCP_SA_KEY_JSON, or env GOOGLE_APPLICATION_CREDENTIALS."
    )


# ───────────────────────────────────────────────────────────────
# 1) One cached Connector + connection per user session
# ───────────────────────────────────────────────────────────────
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
      - ip_type: "PUBLIC" or "PRIVATE" (optional, default PUBLIC)
      - universe_domain: e.g. "googleapis.com" (optional; defaults connector)
    """
    # Always pass explicit credentials to avoid metadata lookups off-GCP
    creds = _load_explicit_credentials()

    connector = Connector(
        credentials=creds,
        # Set if you saw a "universe domain mismatch"; safe default is googleapis.com
        universe_domain=cfg.get("universe_domain", "googleapis.com"),
    )

    def _connect():
        conn = connector.connect(
            cfg["instance_connection_name"],
            "pg8000",
            user=cfg["user"],
            password=cfg["password"],
            db=cfg["db"],
            timeout=10,             # connect timeout (seconds)
            enable_iam_auth=False,  # using DB password auth (not IAM DB Auth)
            ip_type=IPTypes.PRIVATE
            if str(cfg.get("ip_type", "PUBLIC")).upper() == "PRIVATE"
            else IPTypes.PUBLIC,
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
        st.on_session_end(_cleanup)  # no-op on some hosts; guarded above
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
        # Prefer env vars; fallback to Streamlit secrets.
        cloudsql = st.secrets.get("cloudsql", {})
        cfg = {
            "instance_connection_name": os.getenv(
                "INSTANCE_CONNECTION_NAME",
                cloudsql.get("instance_connection_name", ""),
            ),
            "user": os.getenv("DB_USER", cloudsql.get("user", "")),
            # Your password may include a double-quote; keep it raw.
            "password": os.getenv("DB_PASSWORD", cloudsql.get("password", "")),
            "db": os.getenv("DB_NAME", cloudsql.get("db", "")),
            "ip_type": os.getenv("CLOUDSQL_IP_TYPE", cloudsql.get("ip_type", "PUBLIC")),
            "universe_domain": os.getenv(
                "UNIVERSE_DOMAIN",
                cloudsql.get("universe_domain", "googleapis.com"),
            ),
        }

        # Minimal validation
        missing = [k for k in ("instance_connection_name", "user", "password", "db") if not cfg.get(k)]
        if missing:
            raise ValueError(f"Missing DB config values: {', '.join(missing)}")

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
        """Ensure we have a live connection. Quick ping; if it fails, reconnect."""
        try:
            cur = self.conn.cursor()
            try:
                cur.execute("SET LOCAL statement_timeout = 2000;")
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
            finally:
                cur.close()
        except Exception:
            self._reconnect()

    def _fetch_df(self, query: str, params=None) -> pd.DataFrame:
        self._ensure_live_conn()
        try:
            cur = self.conn.cursor()
            try:
                cur.execute("SET LOCAL statement_timeout = 8000;")
                cur.execute(query, params or ())
                rows = cur.fetchall()
                cols = [c[0] for c in cur.description] if cur.description else []
            finally:
                cur.close()
            return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
        except Exception:
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
