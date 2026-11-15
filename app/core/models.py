"""Pydantic contracts shared by the API, graph, UI, and evaluation layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return an ISO-8601 timestamp with an explicit UTC offset."""

    return datetime.now(timezone.utc).isoformat()


class EvidenceItem(BaseModel):
    """A citation that lets an operator reproduce an agent conclusion.

    ``evidence_id`` is the stable citation key; source fields identify where the
    excerpt came from; portfolio/period correlate it to a run; version/effective
    date prevent use of stale policy; ``score`` is retrieval relevance rather
    than factual truth; and ``authoritative`` distinguishes governed sources from
    merely supporting context.
    """
    # Stable key displayed in findings and approval packets.
    evidence_id: str
    # Source type/name tell an auditor where the excerpt originated.
    source_type: str
    source_name: str
    # Scope fields are optional because some policies apply enterprise-wide.
    portfolio_id: str | None = None
    close_period: str | None = None
    effective_date: str | None = None
    version: str | None = None
    section: str | None = None
    # Only the relevant excerpt is propagated; whole documents would add noise.
    excerpt: str
    # Similarity score is 0..1-ish relevance, never an approval probability.
    score: float = 1.0
    authoritative: bool = False


class ExceptionItem(BaseModel):
    """Normalized finding returned by every deterministic finance tool.

    IDs support deduplication/evaluation; category and severity support routing;
    amount/currency and ``materiality_bps`` quantify impact; source refs make the
    calculation reproducible; ``required_approval`` drives the HITL gate rather
    than leaving approval choice to generated prose.
    """
    # ID must be deterministic so repeated agents/runs can deduplicate the issue.
    exception_id: str
    portfolio_id: str
    close_period: str
    category: str
    title: str
    description: str
    # Literal prevents arbitrary severity words that would break routing logic.
    severity: Literal["low", "medium", "high", "critical"]
    amount: float = 0.0
    currency: str = "USD"
    # Basis points normalize financial impact across differently sized funds.
    materiality_bps: float = 0.0
    instrument_id: str | None = None
    status: Literal["open", "investigating", "resolved", "escalated"] = "open"
    source_refs: list[str] = Field(default_factory=list)
    # Tools set this explicit control flag; the LLM cannot waive approval in prose.
    required_approval: bool = False


class AgentOutput(BaseModel):
    """Common envelope returned by local and remote specialist agents.

    ``task_id`` traces one specialist invocation, ``confidence`` represents
    execution completeness, findings are business signals, evidence contains
    citations, recommendations propose next actions, and unresolved questions
    explicitly preserve uncertainty. A shared envelope lets the orchestrator
    treat A2A and in-process agents uniformly.
    """
    agent_name: str
    task_id: str
    summary: str
    # Bounds make confidence safe for UI percentages and escalation thresholds.
    confidence: float = Field(ge=0, le=1)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    # Unknowns remain explicit rather than being hidden in a fluent summary.
    unresolved_questions: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class AgentDecision(BaseModel):
    """Structured supervisor decision used when an LLM selects agents.

    ``selected_agents`` must be registry keys, rationale is safe audit text,
    ``finish`` ends autonomous work, and ``request_human_approval`` routes the
    graph to its interrupt node.
    """
    selected_agents: list[str] = Field(default_factory=list)
    rationale: str
    finish: bool = False
    request_human_approval: bool = False


class ApprovalRequest(BaseModel):
    """Human-in-the-loop packet persisted by ``interrupt``.

    It identifies the run/scope, explains why review is required, names the
    recommendation and supporting evidence, quantifies maximum materiality, and
    later records reviewer identity, decision, comments, and timestamps.
    """
    approval_id: str
    run_id: str
    portfolio_id: str
    reason: str
    recommendation: str
    evidence_ids: list[str] = Field(default_factory=list)
    materiality_bps: float = 0.0
    # Controlled states prevent unsupported approval outcomes.
    status: Literal["pending", "approved", "rejected", "more_information"] = "pending"
    reviewer: str | None = None
    comments: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    decided_at: str | None = None


class RunCreateRequest(BaseModel):
    """Validated input for one review: scenario, portfolio, date, and objective."""
    scenario_id: str
    portfolio_id: str | None = None
    close_period: str = "2026-10-31"
    # Goal gives the supervisor a business objective without exposing tool control.
    user_goal: str = "Complete the month-end review and prepare an approval package."


class ApprovalDecisionRequest(BaseModel):
    """Accountable response used to resume: decision, reviewer, and rationale."""
    decision: Literal["approved", "rejected", "more_information"]
    reviewer: str
    comments: str = ""


class RunSummary(BaseModel):
    """Compact run projection suitable for lists and dashboards."""
    run_id: str
    scenario_id: str
    portfolio_id: str | None
    close_period: str
    status: str
    current_agent: str | None = None
    round_number: int = 0
    exception_count: int = 0
    approval_required: bool = False
    created_at: str
    updated_at: str


class EvaluationResult(BaseModel):
    """Outcome metrics calculated from labels unavailable to agents.

    Counts explain the confusion set; precision measures false-alarm control;
    recall measures missed-risk control; F1 balances them; severity accuracy
    checks prioritization; approval accuracy checks governance routing; and trace
    completeness confirms all expected specialists left auditable evidence.
    """
    run_id: str
    detected_count: int
    expected_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    severity_accuracy: float
    approval_accuracy: float
    trace_completeness: float
    details: list[dict[str, Any]] = Field(default_factory=list)

