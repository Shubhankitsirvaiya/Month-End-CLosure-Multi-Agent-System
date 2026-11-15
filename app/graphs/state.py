"""Typed shared state for the multi-agent month-end review."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_exceptions(existing: list[dict] | None, incoming: list[dict] | None) -> list[dict]:
    """Merge exception updates by stable exception ID.

    Args:
        existing: Findings already present in shared graph state.
        incoming: Findings emitted by the node that just completed.

    Returns:
        One latest record per ``exception_id``. This reducer is essential when
        parallel agents report the same issue: duplicates would inflate counts
        and could incorrectly trigger approval.
    """

    # Keying by exception ID makes a later update replace an older version
    # instead of adding a duplicate finding.
    merged = {item["exception_id"]: item for item in (existing or [])}
    # ``incoming or []`` also handles an empty first update safely.
    for item in incoming or []:
        merged[item["exception_id"]] = item
    # LangGraph state stores findings as a list, so return map values.
    return list(merged.values())


class CloseReviewState(TypedDict, total=False):
    """Supervisor-owned state shared across all orchestration nodes.

    ``total=False`` means nodes may return only the fields they changed.
    ``Annotated[..., reducer]`` tells LangGraph how to combine parallel updates;
    unannotated fields use normal last-write behavior.
    """
    # Immutable request identity and business scope.
    run_id: str
    scenario_id: str
    portfolio_id: str | None
    close_period: str
    user_goal: str
    # Mutable lifecycle fields controlled by the supervisor.
    status: str
    round_number: int

    # Latest supervisor decision and work queue.
    current_plan: str
    pending_agents: list[str]
    # ``operator.add`` appends parallel/iterative history instead of overwriting it.
    called_agents: Annotated[list[str], operator.add]
    completed_agents: Annotated[list[str], operator.add]

    # Findings merge by stable ID; other artifacts append as an immutable audit trail.
    exceptions: Annotated[list[dict[str, Any]], merge_exceptions]
    agent_outputs: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]
    a2a_tasks: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]

    # Governance decision fields populated by critic and approval nodes.
    evidence_complete: bool
    approval_required: bool
    approval_request: dict[str, Any] | None
    approval_decision: dict[str, Any] | None
    # Final business output exists only after routing and approval are resolved.
    final_review_package: dict[str, Any] | None


class SpecialistState(TypedDict, total=False):
    """Private state used inside a specialist's plan-act-synthesize graph."""
    # Input scope supplied by the supervisor.
    run_id: str
    portfolio_id: str | None
    close_period: str
    task: str
    existing_exceptions: list[dict[str, Any]]
    # Plan node outputs.
    selected_tools: list[str]
    planning_rationale: str
    # Execution and synthesis outputs.
    tool_results: list[dict[str, Any]]
    policy_evidence: list[dict[str, Any]]
    output: dict[str, Any]

