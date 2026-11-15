"""Curated agent registry used for safe orchestrator selection."""

from __future__ import annotations

from app.core.config import Settings, get_settings


def get_agent_registry(settings: Settings | None = None) -> dict[str, dict]:
    """Return the supervisor-visible Agent Cards for allowed specialists.

    Args:
        settings: Supplies the remote valuation base URL.

    Returns:
        A name-keyed registry containing descriptions, skills, and deployment
        mode. The LLM may select only these keys; registration is therefore both
        capability discovery and an application-level allow-list.
    """
    settings = settings or get_settings()
    return {
        "reconciliation": {
            "name": "Reconciliation Agent",
            "description": "Finds position, cash, transaction, custody, and settlement breaks.",
            "skills": ["position reconciliation", "cash reconciliation", "duplicate transaction detection"],
            "mode": "local_langgraph",
        },
        "accounting": {
            "name": "Accounting Agent",
            "description": "Investigates ledger, income, fee, accrual, and journal issues.",
            "skills": ["ledger reconciliation", "fee accrual review", "income review", "journal recommendation"],
            "mode": "local_langgraph",
        },
        "risk_policy": {
            "name": "Risk and Policy Agent",
            "description": "Applies mandate, restricted-security, materiality, and approval policies.",
            "skills": ["mandate checks", "materiality", "restricted holdings", "approval routing"],
            "mode": "local_langgraph",
        },
        "valuation": {
            "name": "Remote Valuation Agent",
            "description": "Investigates stale prices, source conflicts, fair-value evidence, and NAV impact.",
            "skills": ["stale-price review", "price challenge", "NAV impact", "fair-value recommendation"],
            "mode": "remote_a2a",
            "agent_card_url": settings.valuation_agent_url.rstrip("/") + "/.well-known/agent-card.json",
        },
        "evidence_critic": {
            "name": "Evidence Critic Agent",
            "description": "Independently checks citations, completeness, ownership, and approval readiness.",
            "skills": ["evidence validation", "citation review", "audit readiness", "open-risk review"],
            "mode": "local_langgraph",
        },
    }

