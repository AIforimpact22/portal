# pages/3_Manage_Connections.py
import datetime as dt
from typing import Iterable, Tuple

import pandas as pd
import streamlit as st
from db_handler import DatabaseManager

st.set_page_config(page_title="Manage Connections", layout="wide")
st.title("Manage Database Connections")

db = DatabaseManager()  # Cloud SQL via pg8000

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────

def _pg8000_err_to_text(e: Exception) -> str:
    try:
        first = e.args[0]
        if isinstance(first, dict):
            code = first.get("C", "")
            detail = first.get("D", "")
            msgtxt = first.get("M", str(e))
            name = first.get("n", "")
            rel = first.get("t", "")
            return f"[{code}] {msgtxt}" + (f" • {detail}" if detail else "") + (f" • table={rel}" if rel else "") + (f" • constraint={name}" if name else "")
    except Exception:
        pass
    return str(e)

@st.cache_data(show_spinner=False, ttl=5)  # keep it fresh; we’re managing live sessions
def load_activity(datname_filter: str | None = None) -> pd.DataFrame:
    q = """
    SELECT
      pid,
      datname,
      usename,
      application_name,
      client_addr,
      backend_type,
      state,
      wait_event_type,
      now() - state_change      AS idle_for,
      xact_start,
      query_start,
      now() - query_start       AS running_for,
      left(regexp_replace(query, '\\s+', ' ', 'g'), 8000) AS query
    FROM pg_stat_activity
    WHERE pid <> pg_backend_pid()
      AND (CASE WHEN %s IS NULL OR %s = '' THEN TRUE ELSE datname = %s END)
    ORDER BY
      state NULLS LAST,
      running_for DESC NULLS LAST,
      idle_for DESC NULLS LAST;
    """
    df = db.fetch_data(q, (datname_filter, datname_filter, datname_filter))
    # Nice display columns
    if not df.empty:
        def _fmt_td(td):
            if pd.isna(td):
                return None
            # Convert PG interval → string
            return str(td)
        df["idle_for"] = df["idle_for"].map(_fmt_td)
        df["running_for"] = df["running_for"].map(_fmt_td)
    return df

def _exec_many(sql: str, params_list: Iterable[Tuple]) -> list[tuple[bool, str]]:
    """
    Execute a statement many times (each with its own params).
    Returns list of (ok, message).
    """
    results: list[tuple[bool, str]] = []
    cur = db.conn.cursor()
    try:
        for p in params_list:
            try:
                cur.execute(sql, p)
                row = cur.fetchone()
                ok = bool(row and row[0])
                msg = "ok" if ok else "not terminated/cancelled"
                results.append((ok, msg))
            except Exception as e:
                results.append((False, _pg8000_err_to_text(e)))
        db.conn.commit()
    except Exception as e:
        db.conn.rollback()
        results.append((False, _pg8000_err_to_text(e)))
    finally:
        cur.close()
    return results

def terminate_pids(pids: list[int]) -> list[tuple[bool, str]]:
    # SELECT pg_terminate_backend(pid)
    return _exec_many("SELECT pg_terminate_backend(%s)", [(int(pid),) for pid in pids])

def cancel_pids(pids: list[int]) -> list[tuple[bool, str]]:
    # SELECT pg_cancel_backend(pid)
    return _exec_many("SELECT pg_cancel_backend(%s)", [(int(pid),) for pid in pids])

@st.cache_data(show_spinner=False, ttl=30)
def list_databases() -> list[str]:
    q = "SELECT datname FROM pg_database WHERE datistemplate = FALSE ORDER BY datname;"
    df = db.fetch_data(q)
    return df["datname"].tolist() if not df.empty else []

# ───────────────────────────────────────────────────────────────
# Filters
# ───────────────────────────────────────────────────────────────

with st.expander("Filters & refresh", expanded=True):
    cols = st.columns(4)
    with cols[0]:
        dbs = [""] + list_databases()
        datname = st.selectbox("Database (optional)", dbs, index=0, help="Filter sessions by database.")
    with cols[1]:
        only_state = st.selectbox(
            "State filter",
            ["(any)", "active", "idle", "idle in transaction", "idle in transaction (aborted)", "disabled"],
            index=0,
            help="Filter by pg_stat_activity.state."
        )
    with cols[2]:
        app_name_like = st.text_input("Application name contains", "", help="Case-insensitive substring.")
    with cols[3]:
        user_like = st.text_input("User contains", "", help="Case-insensitive substring.")

    st.caption("Data refreshes every ~5 seconds while you interact with this page.")

# Load & filter
df = load_activity(datname or None).copy()
if not df.empty:
    if only_state != "(any)":
        df = df[df["state"] == only_state]
    if app_name_like:
        df = df[df["application_name"].str.contains(app_name_like, case=False, na=False)]
    if user_like:
        df = df[df["usename"].str.contains(user_like, case=False, na=False)]

st.subheader("Sessions")
if df.empty:
    st.info("No sessions found (with the current filters).")
else:
    # Show a compact grid
    show_cols = [
        "pid","datname","usename","application_name","client_addr",
        "backend_type","state","running_for","idle_for","query_start","query"
    ]
    st.dataframe(df[show_cols], use_container_width=True, height=420)

# ───────────────────────────────────────────────────────────────
# Actions
# ───────────────────────────────────────────────────────────────

st.subheader("Actions")

with st.expander("Terminate selected PIDs", expanded=True):
    sel_pids = st.multiselect(
        "Choose PIDs to terminate (ROLLBACKs their transactions)",
        options=df["pid"].tolist() if not df.empty else [],
        format_func=lambda x: f"{x}",
    )
    colA, colB = st.columns([1, 2])
    with colA:
        confirm_sel = st.checkbox("I'm sure", value=False)
    with colB:
        st.caption("Terminate forcibly disconnects sessions (use with care).")

    if st.button("Terminate selected", disabled=not sel_pids or not confirm_sel):
        results = terminate_pids(sel_pids)
        ok_count = sum(1 for ok, _ in results if ok)
        st.success(f"Requested terminate on {len(sel_pids)} session(s) • success={ok_count}")
        for pid, (ok, msg) in zip(sel_pids, results):
            st.write(f"- PID {pid}: {'✅' if ok else '❌'} {msg}")
        st.cache_data.clear()

with st.expander("Terminate idle sessions by age", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        min_idle_min = st.number_input("Min idle age (minutes)", min_value=1, max_value=1440, value=15, step=1)
    with col2:
        include_active = st.checkbox("Include active? (No — idle only)", value=False, disabled=True)
    with col3:
        confirm_idle = st.checkbox("I'm sure", value=False)

    # Build candidate list
    candidates = []
    if not df.empty:
        # parse "idle_for" string like '0:07:12.345678'
        def _parse_interval(s: str | None) -> float:
            if not s or not isinstance(s, str):
                return 0.0
            # Fallback parse: HH:MM:SS.micro
            try:
                # Handle days if present "X days HH:MM:SS"
                if "day" in s:
                    parts = s.split("day")
                    days = int(parts[0].strip())
                    rest = parts[1].strip()
                    t = dt.datetime.strptime(rest.split(".")[0], "%H:%M:%S")
                    secs = days * 86400 + t.hour*3600 + t.minute*60 + t.second
                else:
                    t = dt.datetime.strptime(s.split(".")[0], "%H:%M:%S")
                    secs = t.hour*3600 + t.minute*60 + t.second
                return secs / 60.0
            except Exception:
                return 0.0

        for _, r in df.iterrows():
            if str(r.get("state") or "").lower().startswith("idle"):
                idle_min = _parse_interval(r.get("idle_for"))
                if idle_min >= float(min_idle_min):
                    candidates.append(int(r["pid"]))

    st.caption(f"Candidates to terminate (idle ≥ {min_idle_min} min): {len(candidates)}")
    if st.button("Terminate these idle sessions", disabled=(not candidates or not confirm_idle)):
        results = terminate_pids(candidates)
        ok_count = sum(1 for ok, _ in results if ok)
        st.success(f"Requested terminate on {len(candidates)} idle session(s) • success={ok_count}")
        for pid, (ok, msg) in zip(candidates, results):
            st.write(f"- PID {pid}: {'✅' if ok else '❌'} {msg}")
        st.cache_data.clear()

with st.expander("Terminate idle-in-transaction by age", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_iit_min = st.number_input("Min idle-in-transaction age (minutes)", min_value=1, max_value=1440, value=5, step=1)
    with col2:
        confirm_iit = st.checkbox("I'm sure", value=False)

    iit_candidates = []
    if not df.empty:
        def _parse_interval(s: str | None) -> float:
            if not s or not isinstance(s, str):
                return 0.0
            try:
                if "day" in s:
                    parts = s.split("day")
                    days = int(parts[0].strip())
                    rest = parts[1].strip()
                    t = dt.datetime.strptime(rest.split(".")[0], "%H:%M:%S")
                    secs = days * 86400 + t.hour*3600 + t.minute*60 + t.second
                else:
                    t = dt.datetime.strptime(s.split(".")[0], "%H:%M:%S")
                    secs = t.hour*3600 + t.minute*60 + t.second
                return secs / 60.0
            except Exception:
                return 0.0

        for _, r in df.iterrows():
            if str(r.get("state") or "").lower().startswith("idle in transaction"):
                idle_min = _parse_interval(r.get("idle_for"))
                if idle_min >= float(min_iit_min):
                    iit_candidates.append(int(r["pid"]))

    st.caption(f"Candidates to terminate (idle in txn ≥ {min_iit_min} min): {len(iit_candidates)}")
    if st.button("Terminate idle-in-transaction", disabled=(not iit_candidates or not confirm_iit)):
        results = terminate_pids(iit_candidates)
        ok_count = sum(1 for ok, _ in results if ok)
        st.success(f"Requested terminate on {len(iit_candidates)} session(s) • success={ok_count}")
        for pid, (ok, msg) in zip(iit_candidates, results):
            st.write(f"- PID {pid}: {'✅' if ok else '❌'} {msg}")
        st.cache_data.clear()

with st.expander("Cancel long-running active queries", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        min_run_sec = st.number_input("Min running time (seconds)", min_value=5, max_value=86400, value=30, step=5)
    with col2:
        confirm_cancel = st.checkbox("I'm sure", value=False)

    # Build candidates (active + running_for >= N sec)
    active_to_cancel = []
    if not df.empty:
        def _parse_running(s: str | None) -> float:
            if not s or not isinstance(s, str):
                return 0.0
            try:
                if "day" in s:
                    parts = s.split("day")
                    days = int(parts[0].strip())
                    rest = parts[1].strip()
                    t = dt.datetime.strptime(rest.split(".")[0], "%H:%M:%S")
                    secs = days * 86400 + t.hour*3600 + t.minute*60 + t.second
                else:
                    t = dt.datetime.strptime(s.split(".")[0], "%H:%M:%S")
                    secs = t.hour*3600 + t.minute*60 + t.second
                return float(secs)
            except Exception:
                return 0.0

        for _, r in df.iterrows():
            if (r.get("state") == "active") and (_parse_running(r.get("running_for")) >= float(min_run_sec)):
                active_to_cancel.append(int(r["pid"]))

    st.caption(f"Candidates to cancel (active ≥ {min_run_sec} sec): {len(active_to_cancel)}")
    if st.button("Cancel those queries (pg_cancel_backend)", disabled=(not active_to_cancel or not confirm_cancel)):
        results = cancel_pids(active_to_cancel)
        ok_count = sum(1 for ok, _ in results if ok)
        st.success(f"Requested cancel on {len(active_to_cancel)} active query/queries • success={ok_count}")
        for pid, (ok, msg) in zip(active_to_cancel, results):
            st.write(f"- PID {pid}: {'✅' if ok else '❌'} {msg}")
        st.cache_data.clear()

st.caption(
    "Notes: "
    "1) Terminating a session rolls back its open transaction. "
    "2) Cancel only stops the current query (session remains). "
    "3) You may need sufficient privileges (e.g., cloudsqlsuperuser) to signal other backends."
)
