"""Operator UI: submit, watch, approve, inspect evidence, and evaluate."""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ``API_BASE_URL`` is used by Docker Compose; ``STREAMLIT_API_URL`` supports the
# local .env. The final fallback lets a first-time user run everything locally.
API = os.getenv("API_BASE_URL", os.getenv("STREAMLIT_API_URL", "http://localhost:8000")).rstrip("/")
st.set_page_config(page_title="Month-End Review Copilot", page_icon="📊", layout="wide")


def api(method: str, path: str, **kwargs):
    """Call FastAPI and convert transport errors into visible UI feedback.

    Args:
        method: HTTP verb such as ``GET`` or ``POST``.
        path: API-relative path beginning with ``/``.
        **kwargs: Forwarded to ``requests.request``; commonly ``json=...``.

    Returns:
        Parsed JSON on success or ``None`` after rendering an error message.
    """
    try:
        response = requests.request(method, API + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None


def status_badge(status: str) -> str:
    """Translate machine status into a compact operator-facing label."""
    icons = {"completed": "✅", "approval_pending": "🧑‍⚖️", "failed": "❌", "running": "⚙️", "queued": "⏳", "rejected": "⛔"}
    return f"{icons.get(status, '•')} {status.replace('_', ' ').title()}"


st.title("Month-End Portfolio Review Copilot")
st.caption("Dynamic multi-agent investigation • A2A valuation specialist • human approval • evidence-linked audit trail")

# Stop rendering dependent widgets when the backend is unavailable; otherwise
# every subsequent tab would emit another confusing connection error.
health = api("GET", "/health")
if not health:
    st.stop()

with st.sidebar:
    st.subheader("Runtime")
    st.success("FastAPI connected")
    st.write(f"LLM provider: `{health['llm_provider']}`")
    st.write(f"Valuation A2A: `{health['a2a_valuation_url']}`")
    auto_refresh = st.toggle("Auto-refresh active run", value=True)
    if auto_refresh:
        st_autorefresh(interval=1500, key="run-refresh")

# These calls are read-only projections. The UI never imports agent code or reads
# source CSVs directly; FastAPI remains the application boundary.
scenarios = api("GET", "/api/scenarios") or []
runs = api("GET", "/api/runs") or []
tabs = st.tabs(["Start review", "Live operations", "Exceptions", "Approval inbox", "Evidence & A2A", "Evaluation"])

with tabs[0]:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Create a controlled month-end review")
        # Scenario IDs become selectbox values while the complete record remains
        # available for descriptions and default portfolio scope.
        scenario_map = {item.get("scenario_id", item.get("id", f"scenario-{i}")): item for i, item in enumerate(scenarios)}
        scenario_id = st.selectbox("Scenario", list(scenario_map) or ["standard_month_end"])
        chosen = scenario_map.get(scenario_id, {})
        portfolio_options = chosen.get("portfolio_ids") or ([chosen.get("portfolio_id")] if chosen.get("portfolio_id") else [])
        if not portfolio_options:
            profile = api("GET", "/api/dataset/profile") or {}
            portfolio_options = profile.get("portfolio_ids", ["MR-CREDIT-HY"])
        portfolio = st.selectbox("Portfolio", portfolio_options)
        period = st.text_input("Close date", value="2026-10-31")
        goal = st.text_area("Review objective", value="Complete the month-end portfolio review, investigate material exceptions, and prepare an approval-ready package.")
        if st.button("Start agent review", type="primary", use_container_width=True):
            created = api("POST", "/api/runs", json={"scenario_id": scenario_id, "portfolio_id": portfolio, "close_period": period, "user_goal": goal})
            if created:
                st.session_state.active_run_id = created["run_id"]
                st.success(f"Started {created['run_id']}")
                st.rerun()
    with right:
        st.subheader("Why this is agentic")
        st.info("The supervisor sees current state and agent cards, then chooses the next specialist(s). Independent work runs concurrently; agents may be revisited when evidence is incomplete.")
        st.markdown("""
        - Calculations stay deterministic inside governed tools.
        - Specialists are separate LangGraph agents with their own plan–act–synthesize loop.
        - Valuation is independently hosted and reached over A2A.
        - The critic checks evidence before the approval gate.
        - Hidden labels evaluate outcomes without leaking answers to agents.
        """)

# Display labels may change, so the run ID—not the label—is stored in session state.
run_options = {f"{item['run_id']} · {status_badge(item['status'])}": item["run_id"] for item in runs}
default_id = st.session_state.get("active_run_id")
labels = list(run_options)
default_index = next((i for i, label in enumerate(labels) if run_options[label] == default_id), 0)
selected_run_id = run_options[st.selectbox("Active run", labels, index=default_index)] if labels else None
if selected_run_id:
    st.session_state.active_run_id = selected_run_id
run = api("GET", f"/api/runs/{selected_run_id}") if selected_run_id else None
state = (run or {}).get("state") or {}

with tabs[1]:
    if not run:
        st.info("Start a review to see the live agent trace.")
    else:
        st.subheader(status_badge(run["status"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Round", state.get("round_number", 0))
        c2.metric("Agents completed", len(set(state.get("completed_agents", []))))
        c3.metric("Exceptions", len(state.get("exceptions", [])))
        c4.metric("Evidence items", len(state.get("evidence", [])))
        st.markdown(f"**Current supervisor plan:** {state.get('current_plan', 'Waiting for first decision…')}")
        # Events contain concise action summaries and safe facts, never hidden
        # chain-of-thought or secret prompt content.
        events = state.get("events", [])
        if events:
            for event in reversed(events):
                with st.container(border=True):
                    st.markdown(f"**{event.get('agent', 'system').replace('_', ' ').title()}** · `{event.get('event_type')}`")
                    st.write(event.get("message"))
                    st.caption(event.get("timestamp", ""))
        elif run.get("error"):
            st.error(run["error"])

with tabs[2]:
    findings = state.get("exceptions", [])
    if not findings:
        st.info("No exception results are available yet.")
    else:
        frame = pd.DataFrame(findings)
        severity = st.multiselect("Severity", sorted(frame["severity"].unique()), default=sorted(frame["severity"].unique()))
        st.dataframe(frame[frame["severity"].isin(severity)], use_container_width=True, hide_index=True)
        chart = frame.groupby(["category", "severity"], as_index=False).size()
        st.plotly_chart(px.bar(chart, x="category", y="size", color="severity", title="Exception population"), use_container_width=True)

with tabs[3]:
    approval = state.get("approval_request")
    if run and run.get("status") == "approval_pending" and approval:
        st.warning(approval.get("reason"))
        st.write(approval.get("recommendation"))
        st.metric("Maximum materiality", f"{approval.get('materiality_bps', 0):.2f} bps")
        reviewer = st.text_input("Reviewer name", value="Portfolio Control Manager")
        comments = st.text_area("Decision rationale", value="Reviewed the exception register, source evidence, and proposed remediation.")
        decision = st.radio("Decision", ["approved", "more_information", "rejected"], horizontal=True)
        # The button sends identity and rationale together with the decision so
        # the resumed graph creates a useful audit record.
        if st.button("Submit accountable decision", type="primary"):
            result = api("POST", f"/api/runs/{selected_run_id}/approval", json={"decision": decision, "reviewer": reviewer, "comments": comments})
            if result:
                st.success("Decision submitted; the graph is resuming.")
                st.rerun()
    else:
        st.info("No run is currently waiting for human approval.")

with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Evidence used")
        evidence = state.get("evidence", [])
        if evidence:
            st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)
        else:
            st.info("Evidence appears after specialist work completes.")
    with col2:
        st.subheader("A2A boundary")
        card = api("GET", "/api/agents/valuation/card")
        st.json(card)
        if state.get("a2a_tasks"):
            st.subheader("A2A task traces")
            st.json(state["a2a_tasks"])

with tabs[5]:
    st.subheader("Hidden ground-truth evaluation")
    st.caption("Expected exception labels are loaded only here; agents and prompts never receive them.")
    if run and run.get("status") in {"completed", "rejected"}:
        if st.button("Evaluate completed run", type="primary"):
            result = api("POST", f"/api/runs/{selected_run_id}/evaluate")
            if result:
                st.session_state.evaluation = result
        evaluation = st.session_state.get("evaluation") or run.get("evaluation")
        if evaluation:
            cols = st.columns(6)
            for column, key in zip(cols, ["precision", "recall", "f1", "severity_accuracy", "approval_accuracy", "trace_completeness"]):
                column.metric(key.replace("_", " ").title(), f"{evaluation[key] * 100:.1f}%")
            st.dataframe(pd.DataFrame(evaluation["details"]), use_container_width=True, hide_index=True)
    else:
        st.info("Finish or reject a run before evaluating it.")
