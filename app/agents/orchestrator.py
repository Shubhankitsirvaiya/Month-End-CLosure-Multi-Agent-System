"""Dynamic LangGraph orchestrator for the month-end review.

The graph topology is stable, but the orchestrator chooses which specialist
agents to call, which agents can run in parallel, whether additional work is
required, and when a human approval gate is necessary.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.a2a.client import A2AClientError, ValuationA2AClient
from app.agents.registry import get_agent_registry
from app.agents.specialists import build_evidence_critic, build_specialist_agent
from app.core.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.models import AgentDecision, ApprovalRequest, utc_now_iso
from app.graphs.state import CloseReviewState


class OrchestratorSelection(BaseModel):
    """Validated decision emitted by the model-backed supervisor node."""
    selected_agents: list[str] = Field(default_factory=list)
    rationale: str
    finish: bool = False
    request_human_approval: bool = False


def _event(agent: str, event_type: str, message: str, details: dict | None = None) -> dict:
    """Create one operator-visible audit event.

    Args:
        agent: Stable actor name, not a display label.
        event_type: Machine-filterable lifecycle category.
        message: Human-readable summary; never hidden chain-of-thought.
        details: Optional structured facts safe to expose in the UI.
    """
    return {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "event_type": event_type,
        "message": message,
        "details": details or {},
    }


def build_orchestrator_graph(settings: Settings | None = None):
    """Compile the dynamic supervisor graph with an in-memory checkpointer.

    Args:
        settings: Optional runtime configuration for dependency injection.

    Returns:
        A compiled LangGraph. ``MemorySaver`` stores node state by ``thread_id``
        so ``interrupt`` can pause for approval and ``Command(resume=...)`` can
        continue from that exact point.
    """
    settings = settings or get_settings()
    # These dependencies are created once when FastAPI starts, then reused by runs.
    model = get_chat_model(settings)
    registry = get_agent_registry(settings)
    # Valuation appears locally only as a continuity fallback; normal invocation
    # crosses the A2A boundary through ``a2a_client`` below.
    local_agents = {
        "reconciliation": build_specialist_agent("reconciliation", settings),
        "accounting": build_specialist_agent("accounting", settings),
        "risk_policy": build_specialist_agent("risk_policy", settings),
        "valuation": build_specialist_agent("valuation", settings),
        "evidence_critic": build_evidence_critic(settings),
    }
    a2a_client = ValuationA2AClient(settings)

    async def initialize(state: CloseReviewState) -> dict:
        """Initialize only graph-owned fields while preserving request scope."""
        # Return only graph-owned defaults. Request fields such as portfolio_id
        # already exist in state and are preserved by LangGraph.
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
            "events": [_event(
                "orchestrator", "run_started",
                "Month-end review initialized with a bounded agent registry.",
                {"portfolio_id": state.get("portfolio_id"), "scenario_id": state.get("scenario_id")},
            )],
        }

    def mock_selection(state: CloseReviewState) -> OrchestratorSelection:
        """Provide reproducible decisions when no paid model is configured.

        This fallback mirrors sensible dependency ordering but remains state
        aware: it chooses work based on which agents have already completed.
        """
        # A set makes membership checks fast and ignores repeated historical calls.
        called = set(state.get("called_agents", []))
        round_number = state.get("round_number", 0)
        # Reconciliation establishes whether source populations are complete before
        # other specialists rely on positions, cash, or transaction records.
        if "reconciliation" not in called:
            return OrchestratorSelection(
                selected_agents=["reconciliation"],
                rationale="Establish the completeness and reconciliation baseline before relying on specialist conclusions.",
            )
        # These domains are independent after baseline reconciliation, so dispatch
        # all remaining names together and let asyncio run them concurrently.
        core_specialists = [name for name in ["valuation", "accounting", "risk_policy"] if name not in called]
        if core_specialists:
            return OrchestratorSelection(
                selected_agents=core_specialists,
                rationale="Independent valuation, accounting, and policy checks can run in parallel after the baseline reconciliation.",
            )
        # Criticism happens after producing agents so it can evaluate their package.
        if "evidence_critic" not in called:
            return OrchestratorSelection(
                selected_agents=["evidence_critic"],
                rationale="An independent critic must verify evidence and approval readiness before finalization.",
            )
        # One controlled exception is enough to route the combined package to HITL.
        approval_needed = any(item.get("required_approval") for item in state.get("exceptions", []))
        return OrchestratorSelection(
            selected_agents=[],
            rationale="All registered specialist reviews and the evidence critique are complete.",
            finish=True,
            request_human_approval=approval_needed,
        )

    async def orchestrator_decide(state: CloseReviewState) -> dict:
        """Choose the next specialist set or finish/approval action.

        The node receives current shared state on every round. In OpenAI mode it
        supplies compact findings and registered cards to a structured-output
        model. Agent names are filtered against the registry after generation.
        """
        round_number = state.get("round_number", 0) + 1
        # This hard budget prevents autonomous cycles even if model decisions are
        # repetitive or contradictory.
        if round_number > settings.max_orchestrator_rounds:
            approval_needed = any(item.get("required_approval") for item in state.get("exceptions", []))
            selection = OrchestratorSelection(
                rationale="The bounded orchestration round limit was reached; unresolved work requires human review.",
                finish=True,
                request_human_approval=approval_needed or bool(state.get("exceptions")),
            )
        elif model is None:
            selection = mock_selection(state)
        else:
            # Do not offer completed agents unless repeated evidence review is useful.
            available = {
                name: value for name, value in registry.items()
                if name not in set(state.get("called_agents", [])) or name == "evidence_critic"
            }
            # Send only routing-relevant fields to reduce tokens and avoid exposing
            # entire source records to the supervisor model.
            compact_exceptions = [
                {key: item.get(key) for key in ["exception_id", "category", "severity", "materiality_bps", "required_approval"]}
                for item in state.get("exceptions", [])[:80]
            ]
            prompt = f"""You are the controlled month-end close orchestrator.
Goal: {state.get('user_goal')}
Portfolio scope: {state.get('portfolio_id') or 'all portfolios'}
Round: {round_number} of {settings.max_orchestrator_rounds}
Already called: {state.get('called_agents', [])}
Available agent cards: {json.dumps(available)}
Current exceptions: {json.dumps(compact_exceptions)}
Evidence complete: {state.get('evidence_complete', False)}

Select zero or more allowed agents. Use parallel agents when their work is independent. You may call the evidence critic after domain specialists. Finish only when evidence has been reviewed or the round budget is exhausted. Request human approval for material, critical, policy, journal, or unresolved cases. Do not invent agent names."""
            structured = model.with_structured_output(OrchestratorSelection)
            selection = await structured.ainvoke(prompt)
            # Enforce registry authorization after generation. Invalid names vanish.
            selection.selected_agents = [name for name in selection.selected_agents if name in registry]

        event = _event(
            "orchestrator", "agent_selection",
            selection.rationale,
            {"selected_agents": selection.selected_agents, "finish": selection.finish, "round": round_number},
        )
        return {
            "round_number": round_number,
            "current_plan": selection.rationale,
            "pending_agents": selection.selected_agents,
            "approval_required": selection.request_human_approval,
            "status": "approval_pending" if selection.request_human_approval and selection.finish else "running",
            "events": [event],
        }

    async def invoke_local(agent_name: str, state: CloseReviewState) -> tuple[dict, dict | None, dict | None]:
        """Invoke an in-process specialist and normalize the transport tuple."""
        # Specialists receive the minimum state needed for their bounded task, not
        # the supervisor's complete internal history.
        specialist_state = {
            "run_id": state["run_id"],
            "portfolio_id": state.get("portfolio_id"),
            "close_period": state["close_period"],
            "task": f"Perform the {agent_name} review required for this month-end close.",
            "existing_exceptions": state.get("exceptions", []),
        }
        result = await local_agents[agent_name].ainvoke(specialist_state)
        return result["output"], None, None

    async def invoke_valuation(state: CloseReviewState) -> tuple[dict, dict | None, dict | None]:
        """Call valuation over A2A and apply the configured continuity fallback.

        A fallback is recorded as an error/audit fact even when work succeeds;
        otherwise operators could incorrectly believe the remote boundary worked.
        """
        try:
            # Remote result includes both business output and transport trace IDs.
            output, trace = await a2a_client.run_valuation_task(
                run_id=state["run_id"], portfolio_id=state.get("portfolio_id"),
                close_period=state["close_period"],
                task="Investigate all applicable stale, conflicting, or unusual month-end prices.",
                existing_exceptions=state.get("exceptions", []),
            )
            return output, trace, None
        except Exception as exc:
            error = {
                "agent": "valuation", "error_type": type(exc).__name__, "message": str(exc),
                "fallback_used": settings.a2a_fallback_local, "timestamp": utc_now_iso(),
            }
            # Some environments prefer hard failure to fallback; the setting makes
            # that continuity trade-off explicit.
            if not settings.a2a_fallback_local:
                raise
            output, _, _ = await invoke_local("valuation", state)
            output["summary"] = "A2A service was unavailable; local valuation fallback completed. " + output["summary"]
            return output, None, error

    async def dispatch(state: CloseReviewState) -> dict:
        """Run independently selected specialists concurrently.

        ``asyncio.gather`` reduces wall-clock time for I/O-bound agents. With
        ``return_exceptions=True`` one failed specialist becomes an auditable
        result instead of cancelling unrelated work.
        """
        pending = state.get("pending_agents", [])
        if not pending:
            return {"pending_agents": []}

        async def invoke(name: str):
            """Dispatch one selected name to its local or remote invocation path."""
            if name == "valuation":
                return name, await invoke_valuation(state)
            return name, await invoke_local(name, state)

        # Star expansion creates one coroutine per selected agent. gather waits for
        # all of them while allowing I/O waits to overlap.
        results = await asyncio.gather(*(invoke(name) for name in pending), return_exceptions=True)
        outputs: list[dict] = []
        exceptions: list[dict] = []
        evidence: list[dict] = []
        a2a_tasks: list[dict] = []
        errors: list[dict] = []
        events: list[dict] = []
        completed: list[str] = []

        # Normalize successful and failed branches into append-only shared state.
        for result in results:
            if isinstance(result, Exception):
                errors.append({"agent": "unknown", "error_type": type(result).__name__, "message": str(result), "timestamp": utc_now_iso()})
                continue
            name, (output, trace, error) = result
            outputs.append(output)
            completed.append(name)
            # Critic "findings" describe package quality, not business exceptions,
            # so they must not enter the exception register.
            exceptions.extend(output.get("findings", []) if name != "evidence_critic" else [])
            evidence.extend(output.get("evidence", []))
            if trace:
                a2a_tasks.append(trace)
            if error:
                errors.append(error)
            events.append(_event(
                name,
                "agent_completed",
                output.get("summary", f"{name} completed."),
                {"task_id": output.get("task_id"), "confidence": output.get("confidence"), "finding_count": len(output.get("findings", [])), "a2a": bool(trace)},
            ))

        evidence_complete = state.get("evidence_complete", False)
        critic_outputs = [item for item in outputs if item.get("agent_name") == "evidence_critic"]
        if critic_outputs:
            findings = critic_outputs[-1].get("findings", [])
            evidence_complete = bool(findings and findings[0].get("evidence_complete"))

        return {
            "pending_agents": [],
            "called_agents": pending,
            "completed_agents": completed,
            "agent_outputs": outputs,
            "exceptions": exceptions,
            "evidence": evidence,
            "a2a_tasks": a2a_tasks,
            "errors": errors,
            "events": events,
            "evidence_complete": evidence_complete,
        }

    async def approval_gate(state: CloseReviewState) -> dict:
        """Build the review packet and pause graph execution for a human.

        ``interrupt`` serializes the supplied value in the checkpoint. The API
        later resumes this node with a reviewer decision rather than restarting
        the preceding agents.
        """
        # Approval includes explicit flags plus high/critical severity as a defense
        # against an upstream tool forgetting to set the flag.
        high_risk = [item for item in state.get("exceptions", []) if item.get("required_approval") or item.get("severity") in {"high", "critical"}]
        max_bps = max([float(item.get("materiality_bps", 0)) for item in high_risk] or [0.0])
        approval = ApprovalRequest(
            approval_id=f"approval-{uuid.uuid4().hex[:12]}",
            run_id=state["run_id"],
            portfolio_id=state.get("portfolio_id") or "MULTI-PORTFOLIO",
            reason=f"{len(high_risk)} high, critical, or explicitly controlled exceptions require accountable review.",
            recommendation="Approve the prepared review package, reject it, or request additional investigation.",
            evidence_ids=[item.get("evidence_id", "") for item in state.get("evidence", []) if item.get("evidence_id")][:30],
            materiality_bps=max_bps,
        )
        # Execution stops here and the approval packet becomes visible through the
        # checkpoint/API. No lines after this execute until Command(resume=...).
        human_value = interrupt(approval.model_dump())
        decision = dict(human_value)
        decision.setdefault("decided_at", utc_now_iso())
        return {
            "approval_request": approval.model_dump(),
            "approval_decision": decision,
            "status": "running" if decision.get("decision") == "more_information" else decision.get("decision", "rejected"),
            "events": [_event(
                "human_reviewer", "approval_decision",
                f"Reviewer decision: {decision.get('decision')}",
                {"reviewer": decision.get("reviewer"), "comments": decision.get("comments")},
            )],
        }

    async def finalize(state: CloseReviewState) -> dict:
        """Create the evidence-linked business output returned to operators."""
        exceptions = state.get("exceptions", [])
        # Aggregate counts are convenient for dashboards while preserving details.
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
        final_status = "completed"
        if decision.get("decision") == "rejected":
            final_status = "rejected"
        return {
            "final_review_package": package,
            "status": final_status,
            "events": [_event("orchestrator", "run_completed", "Final evidence-linked review package created.", {"exception_count": len(exceptions), "status": final_status})],
        }

    def route_after_decision(state: CloseReviewState) -> Literal["dispatch", "approval_gate", "finalize"]:
        """Translate supervisor state into the next graph edge."""
        if state.get("pending_agents"):
            return "dispatch"
        if state.get("approval_required"):
            return "approval_gate"
        return "finalize"

    def route_after_approval(state: CloseReviewState) -> Literal["orchestrator_decide", "finalize"]:
        """Re-investigate when a reviewer requests information; otherwise finish."""
        if (state.get("approval_decision") or {}).get("decision") == "more_information":
            return "orchestrator_decide"
        return "finalize"

    # Nodes contain behavior; edges describe allowed movement. Conditional edges
    # are what make this a state-driven agent workflow instead of a fixed script.
    graph = StateGraph(CloseReviewState)
    graph.add_node("initialize", initialize)
    graph.add_node("orchestrator_decide", orchestrator_decide)
    graph.add_node("dispatch", dispatch)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "orchestrator_decide")
    graph.add_conditional_edges("orchestrator_decide", route_after_decision)
    graph.add_edge("dispatch", "orchestrator_decide")
    graph.add_conditional_edges("approval_gate", route_after_approval)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=MemorySaver())

