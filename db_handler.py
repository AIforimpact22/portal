import os
import re
import json
import uuid
import base64
import pandas as pd
import streamlit as st

# Cloud SQL Python Connector (PostgreSQL via pg8000)
from google.cloud.sql.connector import Connector, IPTypes
from google.oauth2 import service_account  # explicit creds for off-GCP
import pg8000  # ensure driver is installed


# ───────────────────────────────────────────────────────────────
# Helpers: session key
# ───────────────────────────────────────────────────────────────
def _session_key() -> str:
    """Return a unique key for the current user session."""
    if "_session_key" not in st.session_state:
        st.session_state["_session_key"] = uuid.uuid4().hex
    return st.session_state["_session_key"]


# ───────────────────────────────────────────────────────────────
# PEM cleanup & validation (fixes InvalidByte(..., 61))
# ───────────────────────────────────────────────────────────────
_PEM_HDR = "-----BEGIN PRIVATE KEY-----"
_PEM_FTR = "-----END PRIVATE KEY-----"

def _clean_pem_block(pem: str) -> str:
    """
    Return a normalized PEM string with:
      - exact header/footer lines,
      - no leading spaces on Base64 lines,
      - LF line endings,
      - validated Base64 body length (multiple of 4).
    Raises ValueError with a clear hint if malformed.
    """
    if not isinstance(pem, str) or _PEM_HDR not in pem or _PEM_FTR not in pem:
        raise ValueError("Service account private_key is missing a valid PEM block.")

    # Normalize newlines
    pem = pem.replace("\r\n", "\n").replace("\r", "\n")

    # Extract body
    m = re.search(rf"{re.escape(_PEM_HDR)}\n(.*)\n{re.escape(_PEM_FTR)}", pem, re.S)
    if not m:
        raise ValueError("PEM block must have header, body, and footer on separate lines.")

    body = m.group(1)

    # Remove any leading/trailing spaces on base64 lines
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

    # Join to raw base64 and validate padding/length
    b64 = "".join(lines)
    if len(b64) % 4 != 0:
        # Classic copy/paste corruption; this is what triggers InvalidByte(..., 61)
        raise ValueError(
            f"PEM appears corrupted: Base64 length {len(b64)} is not divisible by 4. "
            "Re-copy the private key from the JSON file (no edits/extra characters)."
        )
    try:
        # Decode to ensure characters are valid. Validate padding as well.
        base64.b64decode(b64, validate=True)
    except Exception as e:
        raise ValueError(f"PEM base64 is invalid: {e}")

    # Reassemble with LF only and a trailing newline (common tooling expects it)
    clean = _PEM_HDR + "\n"
    # Keep 64-char wrapping for readability (optional)
    wrap = 64
    clean += "\n".join(b64[i : i + wrap] for i in range(0, len(b64), wrap)) + "\n"
    clean += _PEM_FTR + "\n"
    return clean


def _normalize_sa_info(info: dict) -> dict:
    """Return a copy of service account info with a cleaned PEM."""
    info = dict(info)  # shallow copy
    if "private_key" in info and isinstance(info["private_key"], str):
        info["private_key"] = _clean_pem_block(info["private_key"])
    return info


# ───────────────────────────────────────────────────────────────
# Credentials loader (accepts Streamlit secrets or envs)
# ───────────────────────────────────────────────────────────────
def _load_explicit_credentials():
    """
    Build service-account Credentials from one of:
      1) st.secrets["gcp_service_account"] (dict)  ← Streamlit preferred
      2) st.secrets["GCP_SA_KEY_JSON"] (string with JSON)
      3) env var GCP_SA_KEY_JSON (string with JSON)
      4) env var GOOGLE_APPLICATION_CREDENTIALS (file path)
    Returns a google.oauth2.service_account.Credentials or raises ValueError.
    """
    try_sources = []

    # 1) Streamlit secret as a dict-like block
    if "gcp_service_account" in st.secrets:
        try_sources.append(("st.secrets['gcp_service_account']", dict(st.secrets["gcp_service_account"])))

    # 2) Streamlit secret JSON string
    if "GCP_SA_KEY_JSON" in st.secrets:
        try:
            try_sources.append(("st.secrets['GCP_SA_KEY_JSON']", json.loads(st.secrets["GCP_SA_KEY_JSON"])))
        except Exception as e:
            raise ValueError(f"GCP_SA_KEY_JSON in secrets is not valid JSON: {e}")

    # 3) Env var JSON string
    if os.getenv("GCP_SA_KEY_JSON"):
        try:
            try_sources.append(("env:GCP_SA_KEY_JSON", json.loads(os.environ["GCP_SA_KEY_JSON"])))
        except Exception as e:
            raise ValueError(f"env GCP_SA_KEY_JSON is not valid JSON: {e}")

    # 4) Env var file path
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        if not os.path.isfile(path):
            raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS points to a non-existent file: {path}")
        # Let the file loader handle PEM correctness
        return service_account.Credentials.from_service_account_file(path)

    # Try JSON/dict sources (with PEM cleanup)
    for label, info in try_sources:
        try:
            info = _normalize_sa_info(info)
            return service_account.Credentials.from_service_account_info(info)
        except ValueError as e:
            # Make PEM errors actionable
            raise ValueError(
                f"Failed to build credentials from {label}: {e}\n"
                "Hint: paste the 'private_key' exactly as issued (use triple quotes in TOML/Streamlit)."
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
    creds = _load_explicit_credentials()

    connector = Connector(
        credentials=creds,
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
        cur = conn.cursor()
        try:
            # Short per-transaction timeouts to keep the UI responsive
            cur.execute("SET statement_timeout = 5000;")
            cur.execute("SET idle_in_transaction_session_timeout = 5000;")
            # Optional: tag connection
            cur.execute("SET application_name = 'streamlit_app';")
        finally:
            cur.close()
        return conn

    conn = _connect()

    # Cache cleanup is handled by Streamlit when the session ends; still guard explicit close.
    def _cleanup():
        try:
            conn.close()
        except Exception:
            pass
        try:
            connector.close()
        except Exception:
            pass

    # Streamlit may not always expose a hook; best-effort attach.
    try:
        st.on_session_end(_cleanup)  # no-op on some hosts
    except Exception:
        pass

    # Keep a handle in case you want manual cleanup elsewhere
    conn._cloudsql_connector = connector  # noqa: SLF001
    return conn


# ───────────────────────────────────────────────────────────────
# 2) Database manager with auto-reconnect logic
# ───────────────────────────────────────────────
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
            # Your password may include % or "; keep it raw.
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
        return self.fetch_data("SELECT supplierid, suppliername FROM supplier")

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
