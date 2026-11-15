"""Modular specialist LangGraph nodes used by API and notebook."""
from __future__ import annotations

import uuid
from functools import partial
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.models import AgentOutput
from app.data.loader import CloseDataBundle, load_close_data
from app.graphs.state import SpecialistState
from app.rag.policy_rag_modular import PolicyRetriever
from app.tools.finance_tools import (
    detect_duplicate_transactions, detect_fee_and_income_exceptions,
    detect_mandate_breaches, detect_stale_and_conflicting_prices,
    reconcile_cash, reconcile_general_ledger, reconcile_positions,
)


ToolFunction = Callable[[CloseDataBundle], list]


class SpecialistPlan(BaseModel):
    """Structured model response constrained to allowed tool names."""

    selected_tools: list[str] = Field(description="Allowed tool names required for this task")
    rationale: str = Field(description="Short auditable reason for the selection")


TOOL_REGISTRY: dict[str, ToolFunction] = {
    "reconcile_positions": reconcile_positions,
    "reconcile_cash": reconcile_cash,
    "detect_duplicate_transactions": detect_duplicate_transactions,
    "detect_stale_and_conflicting_prices": detect_stale_and_conflicting_prices,
    "detect_fee_and_income_exceptions": detect_fee_and_income_exceptions,
    "reconcile_general_ledger": reconcile_general_ledger,
    "detect_mandate_breaches": detect_mandate_breaches,
}


AGENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "reconciliation": {
        "description": "Compares internal, custodian, cash, transaction, and settlement records.",
        "tools": ["reconcile_positions", "reconcile_cash", "detect_duplicate_transactions"],
        "policy_ids": ["REC-001", "SRC-001", "MAT-001"],
    },
    "accounting": {
        "description": "Investigates ledger, income, fee, accrual, and journal issues.",
        "tools": ["detect_fee_and_income_exceptions", "reconcile_general_ledger"],
        "policy_ids": ["FEE-001", "CA-001", "JRN-001", "MAT-001"],
    },
    "risk_policy": {
        "description": "Applies mandate, restricted-security, materiality, and approval policies.",
        "tools": ["detect_mandate_breaches"],
        "policy_ids": ["RISK-001", "MAT-001", "APP-001"],
    },
    "valuation": {
        "description": "Investigates stale/conflicting prices and valuation impact.",
        "tools": ["detect_stale_and_conflicting_prices"],
        "policy_ids": ["VAL-001", "SRC-001", "MAT-001", "APP-001"],
    },
}


async def specialist_plan_node(
    state: SpecialistState,
    *,
    agent_name: str,
    definition: dict[str, Any],
    model: Any,
) -> dict:
    """Select a subset of the specialist's application-enforced tool allow-list."""
    allowed_tools = definition["tools"]
    # The model must make a structured decision from the task and bounded list;
    # configuration fails early when no OpenAI key is available.
    structured = model.with_structured_output(SpecialistPlan)
    response = await structured.ainvoke(f"""You are the {agent_name} month-end specialist.
Task: {state.get('task', '')}
Portfolio: {state.get('portfolio_id') or 'all portfolios'}
Allowed tools: {allowed_tools}
Select only necessary allowed tools and give a two-sentence rationale.""")
    selected = [name for name in response.selected_tools if name in allowed_tools]
    return {"selected_tools": selected or list(allowed_tools), "planning_rationale": response.rationale}


async def execute_tools_node(
    state: SpecialistState,
    *,
    settings: Settings,
) -> dict:
    """Execute deterministic registered calculations for the scoped portfolio."""
    bundle = load_close_data(settings).for_portfolio(state.get("portfolio_id"))
    results = []
    for tool_name in state.get("selected_tools", []):
        for item in TOOL_REGISTRY[tool_name](bundle):
            results.append(item.model_dump())
    # Stable IDs remove overlap between controls without discarding updates.
    unique = {item["exception_id"]: item for item in results}
    return {"tool_results": list(unique.values())}


async def specialist_synthesize_node(
    state: SpecialistState,
    *,
    agent_name: str,
    definition: dict[str, Any],
    retriever: PolicyRetriever,
) -> dict:
    """Attach citable policy and normalize findings into AgentOutput."""
    tool_results = state.get("tool_results", [])
    categories: dict[str, int] = {}
    for item in tool_results:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    evidence = retriever.search(
        query=f"{agent_name} month-end review " + " ".join(categories),
        top_k=5, policy_ids=definition["policy_ids"],
        portfolio_id=state.get("portfolio_id"), close_period=state.get("close_period"),
    )
    high_risk = [item for item in tool_results if item["severity"] in {"high", "critical"}]
    recommendations = []
    if high_risk:
        recommendations.append("Prioritize high and critical exceptions for named-owner review.")
    if any(item.get("required_approval") for item in tool_results):
        recommendations.append("Route the affected decision package through the approval matrix.")
    output = AgentOutput(
        agent_name=agent_name, task_id=f"{agent_name}-{uuid.uuid4().hex[:10]}",
        summary=(f"{agent_name.replace('_', ' ').title()} completed "
                 f"{len(state.get('selected_tools', []))} controlled checks and identified "
                 f"{len(tool_results)} exceptions; {len(high_risk)} are high or critical."),
        confidence=0.96 if tool_results else 0.88,
        findings=tool_results, evidence=evidence,
        recommendations=recommendations or ["No material exception identified in the assigned scope."],
        unresolved_questions=[],
    )
    return {"policy_evidence": [item.model_dump() for item in evidence], "output": output.model_dump()}


def build_specialist_agent(agent_name: str, settings: Settings | None = None):
    """Bind role dependencies to modular nodes and compile one specialist graph."""
    settings = settings or get_settings()
    definition = AGENT_DEFINITIONS[agent_name]
    model = get_chat_model(settings)
    retriever = PolicyRetriever(settings)
    graph = StateGraph(SpecialistState)
    graph.add_node("plan", partial(
        specialist_plan_node, agent_name=agent_name, definition=definition, model=model,
    ))
    graph.add_node("execute_tools", partial(execute_tools_node, settings=settings))
    graph.add_node("synthesize", partial(
        specialist_synthesize_node, agent_name=agent_name,
        definition=definition, retriever=retriever,
    ))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_tools")
    graph.add_edge("execute_tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


async def evidence_review_node(
    state: SpecialistState,
    *,
    retriever: PolicyRetriever,
) -> dict:
    """Check source support and approval flags independently of producers."""
    exceptions = state.get("existing_exceptions", [])
    missing_sources = [item["exception_id"] for item in exceptions if not item.get("source_refs")]
    missing_approval = [
        item["exception_id"] for item in exceptions
        if item.get("severity") in {"high", "critical"} and not item.get("required_approval")
    ]
    evidence = retriever.search(
        "audit evidence approval traceability source version unresolved exception",
        top_k=4, policy_ids=["AUD-001", "APP-001", "SRC-001"],
        portfolio_id=state.get("portfolio_id"), close_period=state.get("close_period"),
    )
    complete = not missing_sources and not missing_approval
    output = AgentOutput(
        agent_name="evidence_critic", task_id=f"evidence-{uuid.uuid4().hex[:10]}",
        summary=(f"Evidence critic reviewed {len(exceptions)} exceptions. Package is "
                 f"{'complete' if complete else 'incomplete'} for approval routing."),
        confidence=0.97 if complete else 0.66,
        findings=[{"evidence_complete": complete,
                   "missing_source_exceptions": missing_sources,
                   "missing_approval_flags": missing_approval}],
        evidence=evidence,
        recommendations=(["Proceed to governed human approval where required."] if complete
                         else ["Return incomplete cases to the responsible specialist."]),
        unresolved_questions=[f"Missing support for {item}" for item in missing_sources],
    )
    return {"policy_evidence": [item.model_dump() for item in evidence], "output": output.model_dump()}


def build_evidence_critic(settings: Settings | None = None):
    """Compile the independent evidence-review graph."""
    settings = settings or get_settings()
    graph = StateGraph(SpecialistState)
    graph.add_node("review_evidence", partial(
        evidence_review_node, retriever=PolicyRetriever(settings),
    ))
    graph.add_edge(START, "review_evidence")
    graph.add_edge("review_evidence", END)
    return graph.compile()
