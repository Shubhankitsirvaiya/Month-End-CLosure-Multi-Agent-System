import json

from app.core.config import get_settings
from app.data.loader import load_close_data
from app.tools.finance_tools import all_finance_checks


def test_deterministic_tools_recover_most_seeded_exceptions():
    """Guard against calculation regressions that reduce exception recall.

    The 85% floor tolerates a few labels designed for agent interpretation while
    requiring deterministic tools to recover the large majority of seeded cases.
    """
    settings = get_settings()
    # Use stable IDs because narrative wording is not a reliable evaluation target.
    detected = {item.exception_id for item in all_finance_checks(load_close_data(settings))}
    payload = json.loads((settings.ground_truth_root / "exceptions.json").read_text(encoding="utf-8"))
    expected_items = payload.get("exceptions", payload) if isinstance(payload, dict) else payload
    expected = {item["exception_id"] for item in expected_items}
    # Intersection = correctly recovered seeded exceptions.
    recall = len(detected & expected) / len(expected)
    assert recall >= 0.85


def test_stable_ids_prevent_duplicate_exception_rows():
    """Ensure parallel/all-tool execution cannot double-count one exception."""
    findings = all_finance_checks(load_close_data())
    # Duplicate IDs would inflate dashboards and approval counts.
    ids = [item.exception_id for item in findings]
    assert len(ids) == len(set(ids))
