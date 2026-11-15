"""Reusable LangGraph specialist-agent builders.

Each specialist has a small plan → execute → synthesize graph. The graph can use
an LLM to select from a bounded tool allowlist, while mock mode makes the same
graph reproducible for tests and classroom demonstrations.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.llm import get_chat_model
from app.core.models import AgentOutput
from app.data.loader import CloseDataBundle, load_close_data
from app.graphs.state import SpecialistState
from app.rag.policy_rag import PolicyRetriever
from app.tools.finance_tools import (
    detect_duplicate_transactions,
    detect_fee_and_income_exceptions,
    detect_mandate_breaches,
    detect_stale_and_conflicting_prices,
    reconcile_cash,
    reconcile_general_ledger,
    reconcile_positions,
)


# Every registered tool accepts the same scoped data bundle and returns findings.
# A common callable signature lets the execution node invoke tools generically.
ToolFunction = Callable[[CloseDataBundle], list]


class SpecialistPlan(BaseModel):
    """Validated LLM planning response restricted to known tool names."""

    selected_tools: list[str] = Field(description="Allowed tool names required for this task")
    rationale: str = Field(description="Short auditable reason for the selection")


# Registry names are the only strings the model may select. Keeping callables in
# application code avoids allowing a generated string to execute arbitrary code.
TOOL_REGISTRY: dict[str, ToolFunction] = {
    "reconcile_positions": reconcile_positions,
    "reconcile_cash": reconcile_cash,
    "detect_duplicate_transactions": detect_duplicate_transactions,
    "detect_stale_and_conflicting_prices": detect_stale_and_conflicting_prices,
    "detect_fee_and_income_exceptions": detect_fee_and_income_exceptions,
    "reconcile_general_ledger": reconcile_general_ledger,
    "detect_mandate_breaches": detect_mandate_breaches,
}


# Each agent gets a purpose, a minimal tool allow-list, and a policy allow-list.
# Those boundaries prove why separate agents are useful: capability and context
# differ even though every specialist uses the same graph factory.
AGENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "reconciliation": {
        "description": "Compares internal, custodian, cash, transaction, and settlement records.",
        "tools": ["reconcile_positions", "reconcile_cash", "detect_duplicate_transactions"],
        "policy_ids": ["REC-001", "SRC-001", "MAT-001"],
    },
    "accounting": {
        "description": "Investigates ledger, income, fee, accrual, and proposed journal issues.",
        "tools": ["detect_fee_and_income_exceptions", "reconcile_general_ledger"],
        "policy_ids": ["FEE-001", "CA-001", "JRN-001", "MAT-001"],
    },
    "risk_policy": {
        "description": "Applies mandate, restricted-security, materiality, and approval policies.",
        "tools": ["detect_mandate_breaches"],
        "policy_ids": ["RISK-001", "MAT-001", "APP-001"],
    },
    "valuation": {
        "description": "Investigates stale and conflicting prices and calculates valuation impact.",
        "tools": ["detect_stale_and_conflicting_prices"],
        "policy_ids": ["VAL-001", "SRC-001", "MAT-001", "APP-001"],
    },
}


def build_specialist_agent(agent_name: str, settings: Settings | None = None):
    """Build one compiled specialist LangGraph.

    Args:
        agent_name: Key in ``AGENT_DEFINITIONS``. It determines the agent's role,
            tool allow-list, and eligible policy documents.
        settings: Optional dependency injection for tests or alternate runtimes.

    Returns:
        A compiled async-capable graph with ``plan``, ``execute_tools``, and
        ``synthesize`` nodes.

    Every specialist follows the same governance pattern, but each receives
    different capabilities. The factory prevents copy/paste drift while still
    producing separate agents and separate state executions.
    """
    settings = settings or get_settings()
    # Direct dictionary access intentionally fails fast for an unregistered role.
    definition = AGENT_DEFINITIONS[agent_name]
    # The same graph supports deterministic and live-model modes.
    model = get_chat_model(settings)
    # Build retrieval once for this compiled agent, not once per graph node.
    retriever = PolicyRetriever(settings)

    async def plan(state: SpecialistState) -> dict:
        """Select tools from this specialist's bounded allow-list.

        The model may choose fewer tools but can never invoke a tool outside the
        allow-list. Mock mode selects all relevant tools for deterministic tests.
        """
        allowed_tools = definition["tools"]
        # Mock mode deliberately selects the complete bounded tool set. This gives
        # reproducible classroom output while exercising the same execute node.
        if model is None:
            return {
                "selected_tools": list(allowed_tools),
                "planning_rationale": (
                    f"Offline demonstration mode selected every bounded {agent_name} tool relevant "
                    "to the assigned month-end scope."
                ),
            }
        # Structured output enforces a Pydantic schema instead of parsing
        # unreliable prose returned by the model.
        structured = model.with_structured_output(SpecialistPlan)
        prompt = f"""You are the {agent_name} specialist in a controlled month-end close.
Task: {state.get('task', '')}
Portfolio: {state.get('portfolio_id') or 'all portfolios'}
Allowed tools: {allowed_tools}
Select only the tools needed. Do not invent tool names. Explain selection in at most two sentences."""
        # ainvoke releases the event loop while waiting for the model API.
        response: SpecialistPlan = await structured.ainvoke(prompt)
        # Enforce the allow-list in application code. A prompt is guidance, not
        # an authorization boundary.
        selected = [tool for tool in response.selected_tools if tool in allowed_tools]
        return {"selected_tools": selected or list(allowed_tools), "planning_rationale": response.rationale}

    async def execute(state: SpecialistState) -> dict:
        """Run the selected deterministic finance calculations.

        LLMs never perform authoritative arithmetic here. Each tool consumes a
        portfolio-scoped source bundle and returns validated ExceptionItems.
        """
        # Load independent source populations, then apply the assigned scope.
        bundle = load_close_data(settings).for_portfolio(state.get("portfolio_id"))
        results: list[dict[str, Any]] = []
        # Tool names came through the allow-list, so dictionary lookup resolves a
        # known callable rather than importing or evaluating generated code.
        for tool_name in state.get("selected_tools", []):
            for item in TOOL_REGISTRY[tool_name](bundle):
                # Convert Pydantic models before storing them in LangGraph state.
                results.append(item.model_dump())
        # Preserve one record per exception even if two tools discover the same case.
        unique = {item["exception_id"]: item for item in results}
        return {"tool_results": list(unique.values())}

    async def synthesize(state: SpecialistState) -> dict:
        """Attach policy evidence and normalize results into ``AgentOutput``.

        Five chunks provide useful coverage without flooding the reviewer.
        ``policy_ids`` constrains retrieval to this specialist's control domain.
        """
        tool_results = state.get("tool_results", [])
        # Categories enrich the retrieval query with the actual problems found.
        category_summary: dict[str, int] = {}
        for item in tool_results:
            category_summary[item["category"]] = category_summary.get(item["category"], 0) + 1
        evidence = retriever.search(
            query=f"{agent_name} month-end review " + " ".join(category_summary),
            top_k=5,
            policy_ids=definition["policy_ids"],
            portfolio_id=state.get("portfolio_id"),
            close_period=state.get("close_period"),
        )
        # High-risk counts make the summary operational rather than purely verbose.
        high_risk = [item for item in tool_results if item["severity"] in {"high", "critical"}]
        summary = (
            f"{agent_name.replace('_', ' ').title()} completed {len(state.get('selected_tools', []))} "
            f"controlled checks and identified {len(tool_results)} exceptions; {len(high_risk)} are high or critical."
        )
        # Recommendations are rule-based here so safety routing does not depend on
        # whether the model happened to mention approval in natural language.
        recommendations = []
        if high_risk:
            recommendations.append("Prioritize high and critical exceptions for named-owner review.")
        if any(item.get("required_approval") for item in tool_results):
            recommendations.append("Route the affected decision package through the approval matrix.")
        output = AgentOutput(
            agent_name=agent_name,
            task_id=f"{agent_name}-{uuid.uuid4().hex[:10]}",
            summary=summary,
            # Confidence describes execution completeness, not investment truth.
            confidence=0.96 if tool_results else 0.88,
            findings=tool_results,
            evidence=evidence,
            recommendations=recommendations or ["No material exception identified in the assigned scope."],
            unresolved_questions=[],
        )
        # Return only state updates for this node; LangGraph merges them.
        return {"policy_evidence": [item.model_dump() for item in evidence], "output": output.model_dump()}

    # The topology stays small enough to teach while the planning node still
    # chooses actions dynamically at runtime.
    graph = StateGraph(SpecialistState)
    graph.add_node("plan", plan)
    graph.add_node("execute_tools", execute)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute_tools")
    graph.add_edge("execute_tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def build_evidence_critic(settings: Settings | None = None):
    """Build an independent critic that checks approval readiness.

    The critic does not rediscover business exceptions. It verifies source
    references and approval flags so the producing agent does not grade itself.
    """
    settings = settings or get_settings()
    retriever = PolicyRetriever(settings)

    async def review(state: SpecialistState) -> dict:
        """Inspect exception support and return a completeness verdict."""
        exceptions = state.get("existing_exceptions", [])
        # A finding without source references cannot be reproduced by a reviewer.
        missing_sources = [item["exception_id"] for item in exceptions if not item.get("source_refs")]
        # High/critical findings must carry explicit approval routing metadata.
        missing_approval = [
            item["exception_id"] for item in exceptions
            if item.get("severity") in {"high", "critical"} and not item.get("required_approval")
        ]
        evidence = retriever.search(
            "audit evidence approval traceability source version unresolved exception",
            top_k=4,
            policy_ids=["AUD-001", "APP-001", "SRC-001"],
            portfolio_id=state.get("portfolio_id"),
            close_period=state.get("close_period"),
        )
        complete = not missing_sources and not missing_approval
        output = AgentOutput(
            agent_name="evidence_critic",
            task_id=f"evidence-{uuid.uuid4().hex[:10]}",
            summary=(
                f"Evidence critic reviewed {len(exceptions)} exceptions. "
                f"Package is {'complete' if complete else 'incomplete'} for approval routing."
            ),
            confidence=0.97 if complete else 0.66,
            findings=[{
                "evidence_complete": complete,
                "missing_source_exceptions": missing_sources,
                "missing_approval_flags": missing_approval,
            }],
            evidence=evidence,
            recommendations=(
                ["Proceed to governed human approval where required."]
                if complete
                else ["Return incomplete cases to the responsible specialist before approval."]
            ),
            unresolved_questions=[f"Missing support for {item}" for item in missing_sources],
        )
        return {"policy_evidence": [item.model_dump() for item in evidence], "output": output.model_dump()}

    graph = StateGraph(SpecialistState)
    graph.add_node("review_evidence", review)
    graph.add_edge(START, "review_evidence")
    graph.add_edge("review_evidence", END)
    return graph.compile()

