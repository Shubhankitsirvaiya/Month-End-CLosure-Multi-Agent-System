"""Modular dynamic LangGraph orchestrator used by the API and notebook.

Each node is top-level so it can be tested, taught, and executed independently.
``OrchestratorRuntime`` carries dependencies that were previously captured by
nested closures. The graph behavior and contracts remain the same.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.a2a.client_modular import ValuationA2AClient
from app.agents.registry import get_agent_registry
from app.agents.specialists_modular import build_evidence_critic, build_specialist_agent
from app.core.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.models import ApprovalRequest, utc_now_iso
from app.graphs.state import CloseReviewState


class OrchestratorSelection(BaseModel):
    """Validated output of the model-backed supervisor decision."""

    selected_agents: list[str] = Field(default_factory=list)
    rationale: str
    finish: bool = False
    request_human_approval: bool = False


@dataclass(slots=True)
class OrchestratorRuntime:
    """Dependencies shared by graph nodes but not serialized into graph state."""

    settings: Settings
    model: Any
    registry: dict[str, dict]
    local_agents: dict[str, Any]
    a2a_client: ValuationA2AClient


def create_runtime(settings: Settings | None = None) -> OrchestratorRuntime:
    """Construct reusable model, registry, specialist, and A2A dependencies."""
    # Resolve settings once so every node uses the same run budgets and endpoints.
    settings = settings or get_settings()
    # The required live model makes structured decisions from current shared state.
    model = get_chat_model(settings)
    registry = get_agent_registry(settings)
    # Local valuation is retained only for the configured A2A continuity fallback.
    local_agents = {
        "reconciliation": build_specialist_agent("reconciliation", settings),
        "accounting": build_specialist_agent("accounting", settings),
        "risk_policy": build_specialist_agent("risk_policy", settings),
        "valuation": build_specialist_agent("valuation", settings),
        "evidence_critic": build_evidence_critic(settings),
    }
    return OrchestratorRuntime(
        settings=settings,
        model=model,
        registry=registry,
        local_agents=local_agents,
        a2a_client=ValuationA2AClient(settings),
    )


def make_event(agent: str, event_type: str, message: str, details: dict | None = None) -> dict:
    """Create one safe operator event instead of exposing chain-of-thought."""
    return {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event_type": event_type,
        "message": message,
        "details": details or {},
    }


async def initialize_node(state: CloseReviewState) -> dict:
    """Initialize graph-owned collections and lifecycle fields."""
    # Request fields already exist in state; return only defaults owned by graph.
    return {
        "status": "running",
        "round_number": 0,
        "current_plan": "Inspect the close scope and delegate bounded investigations.",
        "pending_agents": [],
        "exceptions": [],
        "agent_outputs": [],
        "evidence": [],
        "called_agents": [],
        "completed_agents": [],
        "a2a_tasks": [],
        "errors": [],
        "evidence_complete": False,
        "approval_required": False,
        "approval_request": None,
        "approval_decision": None,
        "events": [make_event(
            "orchestrator", "run_started",
            "Month-end review initialized with a bounded agent registry.",
            {"portfolio_id": state.get("portfolio_id"), "scenario_id": state.get("scenario_id")},
        )],
    }


async def decide_node(state: CloseReviewState, *, deps: OrchestratorRuntime) -> dict:
    """Choose the next registered agents or route toward completion."""
    round_number = state.get("round_number", 0) + 1
    # The round budget is a hard stop against autonomous loops.
    if round_number > deps.settings.max_orchestrator_rounds:
        approval_needed = any(item.get("required_approval") for item in state.get("exceptions", []))
        selection = OrchestratorSelection(
            rationale="The bounded orchestration round limit was reached.",
            finish=True,
            request_human_approval=approval_needed or bool(state.get("exceptions")),
        )
    else:
        # Offer only allowed cards and compact routing facts to the supervisor.
        called = set(state.get("called_agents", []))
        available = {
            name: value for name, value in deps.registry.items()
            if name not in called or name == "evidence_critic"
        }
        compact_exceptions = [
            {key: item.get(key) for key in ["exception_id", "category", "severity", "materiality_bps", "required_approval"]}
            for item in state.get("exceptions", [])[:80]
        ]
        prompt = f"""You are the controlled month-end close orchestrator.
Goal: {state.get('user_goal')}
Portfolio: {state.get('portfolio_id') or 'all portfolios'}
Round: {round_number} of {deps.settings.max_orchestrator_rounds}
Already called: {state.get('called_agents', [])}
Available agent cards: {json.dumps(available)}
Current exceptions: {json.dumps(compact_exceptions)}
Evidence complete: {state.get('evidence_complete', False)}

Select only registered agents. Parallelize independent work. Finish only after evidence review. Request human approval for material, critical, policy, journal, or unresolved cases."""
        structured = deps.model.with_structured_output(OrchestratorSelection)
        selection = await structured.ainvoke(prompt)
        # Application code—not the prompt—enforces the registry allow-list.
        selection.selected_agents = [name for name in selection.selected_agents if name in deps.registry]

    return {
        "round_number": round_number,
        "current_plan": selection.rationale,
        "pending_agents": selection.selected_agents,
        "approval_required": selection.request_human_approval,
        "status": "approval_pending" if selection.request_human_approval and selection.finish else "running",
        "events": [make_event(
            "orchestrator", "agent_selection", selection.rationale,
            {"selected_agents": selection.selected_agents, "finish": selection.finish, "round": round_number},
        )],
    }


async def invoke_local_agent(
    agent_name: str,
    state: CloseReviewState,
    deps: OrchestratorRuntime,
) -> tuple[dict, dict | None, dict | None]:
    """Invoke one in-process specialist using the common output tuple."""
    specialist_state = {
        "run_id": state["run_id"],
        "portfolio_id": state.get("portfolio_id"),
        "close_period": state["close_period"],
        "task": f"Perform the {agent_name} review required for this month-end close.",
        "existing_exceptions": state.get("exceptions", []),
    }
    result = await deps.local_agents[agent_name].ainvoke(specialist_state)
    return result["output"], None, None


async def invoke_valuation_agent(
    state: CloseReviewState,
    deps: OrchestratorRuntime,
) -> tuple[dict, dict | None, dict | None]:
    """Call valuation over A2A and use the configured continuity fallback."""
    try:
        output, trace = await deps.a2a_client.run_valuation_task(
            run_id=state["run_id"],
            portfolio_id=state.get("portfolio_id"),
            close_period=state["close_period"],
            task="Investigate all applicable stale, conflicting, or unusual month-end prices.",
            existing_exceptions=state.get("exceptions", []),
        )
        return output, trace, None
    except Exception as exc:
        # Record fallback as an operational error even when local work succeeds.
        error = {
            "agent": "valuation",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "fallback_used": deps.settings.a2a_fallback_local,
            "timestamp": utc_now_iso(),
        }
        if not deps.settings.a2a_fallback_local:
            raise
        output, _, _ = await invoke_local_agent("valuation", state, deps)
        output["summary"] = "A2A service was unavailable; local valuation fallback completed. " + output["summary"]
        return output, None, error


async def dispatch_node(state: CloseReviewState, *, deps: OrchestratorRuntime) -> dict:
    """Run selected specialists concurrently and normalize their updates."""
    pending = state.get("pending_agents", [])
    if not pending:
        return {"pending_agents": []}

    async def invoke(name: str):
        """Choose remote valuation or normal local invocation for one name."""
        if name == "valuation":
            return name, await invoke_valuation_agent(state, deps)
        return name, await invoke_local_agent(name, state, deps)

    # return_exceptions keeps one failure from cancelling independent specialists.
    results = await asyncio.gather(*(invoke(name) for name in pending), return_exceptions=True)
    outputs, exceptions, evidence = [], [], []
    a2a_tasks, errors, events, completed = [], [], [], []

    for result in results:
        if isinstance(result, Exception):
            errors.append({
                "agent": "unknown", "error_type": type(result).__name__,
                "message": str(result), "timestamp": utc_now_iso(),
            })
            continue
        name, (output, trace, error) = result
        outputs.append(output)
        completed.append(name)
        # Critic quality findings do not belong in the business exception register.
        exceptions.extend(output.get("findings", []) if name != "evidence_critic" else [])
        evidence.extend(output.get("evidence", []))
        if trace:
            a2a_tasks.append(trace)
        if error:
            errors.append(error)
        events.append(make_event(
            name, "agent_completed", output.get("summary", f"{name} completed."),
            {"task_id": output.get("task_id"), "confidence": output.get("confidence"),
             "finding_count": len(output.get("findings", [])), "a2a": bool(trace)},
        ))

    evidence_complete = state.get("evidence_complete", False)
    critic_outputs = [item for item in outputs if item.get("agent_name") == "evidence_critic"]
    if critic_outputs:
        critic_findings = critic_outputs[-1].get("findings", [])
        evidence_complete = bool(critic_findings and critic_findings[0].get("evidence_complete"))

    return {
        "pending_agents": [], "called_agents": pending,
        "completed_agents": completed, "agent_outputs": outputs,
        "exceptions": exceptions, "evidence": evidence,
        "a2a_tasks": a2a_tasks, "errors": errors,
        "events": events, "evidence_complete": evidence_complete,
    }


async def approval_node(state: CloseReviewState) -> dict:
    """Pause on material/controlled findings and receive a reviewer decision."""
    high_risk = [
        item for item in state.get("exceptions", [])
        if item.get("required_approval") or item.get("severity") in {"high", "critical"}
    ]
    max_bps = max([float(item.get("materiality_bps", 0)) for item in high_risk] or [0.0])
    approval = ApprovalRequest(
        approval_id=f"approval-{uuid.uuid4().hex[:12]}",
        run_id=state["run_id"],
        portfolio_id=state.get("portfolio_id") or "MULTI-PORTFOLIO",
        reason=f"{len(high_risk)} high, critical, or explicitly controlled exceptions require accountable review.",
        recommendation="Approve, reject, or request additional investigation.",
        evidence_ids=[item.get("evidence_id", "") for item in state.get("evidence", []) if item.get("evidence_id")][:30],
        materiality_bps=max_bps,
    )
    # Nothing below runs until the API supplies Command(resume=decision).
    human_value = interrupt(approval.model_dump())
    decision = dict(human_value)
    decision.setdefault("decided_at", utc_now_iso())
    return {
        "approval_request": approval.model_dump(),
        "approval_decision": decision,
        "status": "running" if decision.get("decision") == "more_information" else decision.get("decision", "rejected"),
        "events": [make_event(
            "human_reviewer", "approval_decision",
            f"Reviewer decision: {decision.get('decision')}",
            {"reviewer": decision.get("reviewer"), "comments": decision.get("comments")},
        )],
    }


async def finalize_node(state: CloseReviewState) -> dict:
    """Create the evidence-linked final business package."""
    exceptions = state.get("exceptions", [])
    severity_counts: dict[str, int] = {}
    for item in exceptions:
        severity_counts[item["severity"]] = severity_counts.get(item["severity"], 0) + 1
    decision = state.get("approval_decision") or {}
    package = {
        "run_id": state["run_id"],
        "portfolio_id": state.get("portfolio_id") or "MULTI-PORTFOLIO",
        "close_period": state["close_period"],
        "executive_summary": (
            f"The multi-agent review identified {len(exceptions)} controlled exceptions across "
            f"{len(set(state.get('completed_agents', [])))} specialist agents."
        ),
        "severity_counts": severity_counts,
        "exceptions": exceptions,
        "agent_summaries": [
            {"agent": item.get("agent_name"), "summary": item.get("summary"), "confidence": item.get("confidence")}
            for item in state.get("agent_outputs", [])
        ],
        "evidence_complete": state.get("evidence_complete", False),
        "approval": decision,
        "open_risks": [item for item in exceptions if item.get("severity") in {"high", "critical"}],
        "generated_at": utc_now_iso(),
    }
    final_status = "rejected" if decision.get("decision") == "rejected" else "completed"
    return {
        "final_review_package": package,
        "status": final_status,
        "events": [make_event(
            "orchestrator", "run_completed",
            "Final evidence-linked review package created.",
            {"exception_count": len(exceptions), "status": final_status},
        )],
    }


def route_after_decision(state: CloseReviewState) -> Literal["dispatch", "approval", "finalize"]:
    """Route from supervisor decision to work, human review, or completion."""
    if state.get("pending_agents"):
        return "dispatch"
    if state.get("approval_required"):
        return "approval"
    return "finalize"


def route_after_approval(state: CloseReviewState) -> Literal["decide", "finalize"]:
    """Return to investigation only when the reviewer requests more information."""
    if (state.get("approval_decision") or {}).get("decision") == "more_information":
        return "decide"
    return "finalize"


def build_orchestrator_graph(settings: Settings | None = None):
    """Wire modular nodes into the same dynamic checkpointed workflow."""
    deps = create_runtime(settings)
    graph = StateGraph(CloseReviewState)
    graph.add_node("initialize", initialize_node)
    # partial binds non-state dependencies while preserving one state input.
    graph.add_node("decide", partial(decide_node, deps=deps))
    graph.add_node("dispatch", partial(dispatch_node, deps=deps))
    graph.add_node("approval", approval_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "decide")
    graph.add_conditional_edges("decide", route_after_decision)
    graph.add_edge("dispatch", "decide")
    graph.add_conditional_edges("approval", route_after_approval)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())
