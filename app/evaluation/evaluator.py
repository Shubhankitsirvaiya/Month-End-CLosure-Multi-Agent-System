"""Evaluate a completed run against labels hidden from every agent.

This is intentionally an outcome evaluation, not a comparison of generated
wording. Stable exception IDs let us measure detection precision and recall.
"""
from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings, get_settings
from app.core.models import EvaluationResult


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide safely and round metrics to four decimal places."""
    return round(numerator / denominator, 4) if denominator else 0.0


def evaluate_run(run: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    """Compare detected findings with hidden expected exception IDs.

    Args:
        run: Persisted run record containing final shared graph state.
        settings: Locates the ground-truth fixture.

    Returns:
        Precision, recall, F1, severity accuracy, approval accuracy, trace
        completeness, and per-exception outcomes.

    Ground truth is loaded only here. Passing it into agent state would leak the
    answers and make the evaluation meaningless.
    """
    settings = settings or get_settings()
    # This path is intentionally absent from prompts, state initialization, and tools.
    truth_path = settings.ground_truth_root / "exceptions.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    expected_items = truth.get("exceptions", truth) if isinstance(truth, dict) else truth
    portfolio_id = run.get("portfolio_id")
    # Compare only labels inside the requested business scope.
    if portfolio_id:
        expected_items = [item for item in expected_items if item.get("portfolio_id") == portfolio_id]

    state = run.get("state") or {}
    detected_items = state.get("exceptions", [])
    expected = {item["exception_id"]: item for item in expected_items}
    detected = {item["exception_id"]: item for item in detected_items}
    # Set algebra gives transparent TP/FP/FN populations by stable ID.
    expected_ids, detected_ids = set(expected), set(detected)
    true_ids = expected_ids & detected_ids
    false_positive_ids = detected_ids - expected_ids
    false_negative_ids = expected_ids - detected_ids

    # Precision answers "how many alerts were real?"; recall answers "how many
    # seeded risks were found?"; F1 balances the two.
    precision = _safe_ratio(len(true_ids), len(detected_ids))
    recall = _safe_ratio(len(true_ids), len(expected_ids))
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    severity_correct = sum(
        detected[item_id].get("severity") == expected[item_id].get("severity") for item_id in true_ids
    )
    severity_accuracy = _safe_ratio(severity_correct, len(true_ids))

    # Approval routing is evaluated independently of wording quality.
    expected_approval = any(item.get("required_approval", False) for item in expected_items)
    actual_approval = bool(state.get("approval_required") or state.get("approval_request"))
    approval_accuracy = 1.0 if expected_approval == actual_approval else 0.0
    completed = set(state.get("completed_agents", []))
    required_trace = {"reconciliation", "accounting", "risk_policy", "valuation", "evidence_critic"}
    trace_completeness = _safe_ratio(len(completed & required_trace), len(required_trace))

    details = []
    for item_id in sorted(expected_ids | detected_ids):
        details.append({
            "exception_id": item_id,
            "outcome": "true_positive" if item_id in true_ids else "false_positive" if item_id in false_positive_ids else "false_negative",
            "expected_severity": expected.get(item_id, {}).get("severity"),
            "detected_severity": detected.get(item_id, {}).get("severity"),
        })

    result = EvaluationResult(
        run_id=run["run_id"], detected_count=len(detected_ids), expected_count=len(expected_ids),
        true_positives=len(true_ids), false_positives=len(false_positive_ids), false_negatives=len(false_negative_ids),
        precision=precision, recall=recall, f1=f1, severity_accuracy=severity_accuracy,
        approval_accuracy=approval_accuracy, trace_completeness=trace_completeness, details=details,
    )
    return result.model_dump()
