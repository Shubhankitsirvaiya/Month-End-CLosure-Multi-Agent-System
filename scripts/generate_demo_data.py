"""Generate a deterministic, internally consistent month-end finance universe.

The data is synthetic but intentionally shaped like a real asset-management
close. A clean accounting population is generated first. Controlled exceptions
are then injected into independent source systems. Every injected issue is
recorded in ``data/ground_truth`` so the application can be evaluated without
leaking answers into the agents or policy index.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


# Fixed seed makes quantities and IDs reproducible for tests and classroom output.
SEED = 16033
# A fixed close date prevents stale-price results changing with today's date.
CLOSE_DATE = date(2026, 10, 31)
# Resolve paths from this script, not from whichever folder launched Python.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GENERATED = DATA / "generated"
POLICIES = DATA / "policies"
GROUND_TRUTH = DATA / "ground_truth"
SCENARIOS = DATA / "scenarios"


# Portfolios intentionally differ in strategy, currency, NAV, and mandate limits
# so the same nominal issue can have different materiality and policy outcomes.
PORTFOLIOS = [
    {"portfolio_id": "MR-EQ-INDIA", "name": "Meridian India Equity Fund", "strategy": "India large and mid-cap equity", "base_currency": "INR", "nav": 185_000_000, "concentration_limit_pct": 40.0},
    {"portfolio_id": "MR-EQ-GLOBAL", "name": "Meridian Global Equity Fund", "strategy": "Developed-market global equity", "base_currency": "USD", "nav": 62_000_000, "concentration_limit_pct": 35.0},
    {"portfolio_id": "MR-BOND-GOV", "name": "Meridian Sovereign Bond Fund", "strategy": "Government and agency debt", "base_currency": "INR", "nav": 240_000_000, "concentration_limit_pct": 45.0},
    {"portfolio_id": "MR-CREDIT-IG", "name": "Meridian Investment Grade Credit Fund", "strategy": "Investment-grade corporate bonds", "base_currency": "USD", "nav": 91_000_000, "concentration_limit_pct": 35.0},
    {"portfolio_id": "MR-CREDIT-HY", "name": "Meridian Opportunistic Credit Fund", "strategy": "High-yield and distressed credit", "base_currency": "USD", "nav": 48_000_000, "concentration_limit_pct": 35.0},
    {"portfolio_id": "MR-BALANCED", "name": "Meridian Balanced Allocation Fund", "strategy": "Multi-asset balanced allocation", "base_currency": "USD", "nav": 125_000_000, "concentration_limit_pct": 35.0},
    {"portfolio_id": "MR-INCOME-MC", "name": "Meridian Multi-Currency Income Fund", "strategy": "Global income with currency overlays", "base_currency": "USD", "nav": 76_000_000, "concentration_limit_pct": 35.0},
    {"portfolio_id": "MR-ESG", "name": "Meridian Sustainable Leaders Fund", "strategy": "ESG-screened global equities", "base_currency": "USD", "nav": 54_000_000, "concentration_limit_pct": 15.0},
]


# Tuple order: ID, issuer, currency, reference price, sector, restricted flag.
EQUITIES = [
    ("EQ001", "Asterion Technologies", "INR", 1840.0, "Technology", False),
    ("EQ002", "Bharat Consumer Products", "INR", 712.0, "Consumer", False),
    ("EQ003", "Crescent Financial Services", "INR", 426.0, "Financials", False),
    ("EQ004", "Delta Infrastructure", "INR", 338.0, "Industrials", False),
    ("EQ005", "Eastern Health Sciences", "INR", 1295.0, "Healthcare", False),
    ("EQ006", "Fjord Semiconductor", "USD", 142.5, "Technology", False),
    ("EQ007", "Greenhaven Utilities", "USD", 58.2, "Utilities", False),
    ("EQ008", "Harbor Retail Group", "USD", 91.4, "Consumer", False),
    ("EQ009", "Ionis Medical Devices", "USD", 117.3, "Healthcare", False),
    ("EQ010", "Juniper Cloud Systems", "USD", 204.8, "Technology", False),
    ("EQ011", "Kestrel Renewable Energy", "USD", 47.6, "Energy", False),
    ("EQ012", "Lattice Thermal Coal", "USD", 36.9, "Energy", True),
    ("EQ013", "Meridian Diversified Holdings", "USD", 83.7, "Multi-Asset", False),
]


# Bond tuples additionally carry coupon, maturity, and credit rating.
BONDS = [
    ("BD001", "Republic of India 7.18% 2033", "INR", 101.2, 7.18, "2033-08-14", "Government", "AAA", False),
    ("BD002", "National Infrastructure 7.65% 2031", "INR", 99.4, 7.65, "2031-03-20", "Agency", "AAA", False),
    ("BD003", "State Development Loan 7.42% 2034", "INR", 98.8, 7.42, "2034-06-10", "Government", "AA+", False),
    ("BD004", "Republic Treasury 6.91% 2029", "INR", 100.5, 6.91, "2029-11-02", "Government", "AAA", False),
    ("BD005", "Arbor Telecom 4.25% 2030", "USD", 97.6, 4.25, "2030-05-18", "Corporate", "A", False),
    ("BD006", "Bluewater Logistics 5.10% 2032", "USD", 94.3, 5.10, "2032-09-12", "Corporate", "BBB+", False),
    ("BD007", "Cobalt Healthcare 4.80% 2028", "USD", 102.1, 4.80, "2028-01-22", "Corporate", "A-", False),
    ("BD008", "Driftwood Packaging 6.90% 2031", "USD", 91.8, 6.90, "2031-07-15", "High Yield", "BB", False),
    ("BD009", "Everpeak Mining 8.25% 2029", "USD", 84.5, 8.25, "2029-12-01", "High Yield", "B+", False),
    ("BD010", "Frontier Aviation 9.10% 2030", "USD", 78.9, 9.10, "2030-04-19", "Distressed", "B-", False),
    ("BD011", "Granite Property Finance 7.40% 2028", "EUR", 96.7, 7.40, "2028-10-05", "High Yield", "BB-", False),
    ("BD012", "Helios Sanctioned Holdings 10.0% 2027", "USD", 66.2, 10.0, "2027-02-28", "Restricted", "CCC", True),
    ("BD013", "Ironclad Finance 4.70% 2031", "USD", 98.1, 4.70, "2031-11-17", "Corporate", "A-", False),
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write dictionaries to CSV with stable column order.

    Args:
        path: Destination file, whose parent directory is created if needed.
        rows: Record dictionaries representing one source-system population.
        fieldnames: Optional explicit schema for an empty or strictly ordered file.
    """
    # parents=True creates nested data folders; exist_ok makes regeneration safe.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty dataset to {path}")
    # Default to insertion order from the first record unless schema is explicit.
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        # DictWriter preserves a readable schema and quotes values when necessary.
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write human-readable UTF-8 JSON for scenarios, profiles, and labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_security_master() -> list[dict[str, Any]]:
    """Create varied equities and bonds with sectors, currencies, and controls."""
    rows: list[dict[str, Any]] = []
    # Normalize equity and bond tuples into one security-master schema.
    for security_id, name, currency, price, sector, restricted in EQUITIES:
        rows.append({
            "security_id": security_id,
            "security_name": name,
            "asset_class": "Equity",
            "currency": currency,
            "sector": sector,
            "coupon_rate": "",
            "maturity_date": "",
            "credit_rating": "",
            "reference_price": price,
            # CSV has no boolean type, so store an explicit lowercase representation.
            "restricted": str(restricted).lower(),
        })
    for security_id, name, currency, price, coupon, maturity, sector, rating, restricted in BONDS:
        rows.append({
            "security_id": security_id,
            "security_name": name,
            "asset_class": "Bond",
            "currency": currency,
            "sector": sector,
            "coupon_rate": coupon,
            "maturity_date": maturity,
            "credit_rating": rating,
            "reference_price": price,
            "restricted": str(restricted).lower(),
        })
    return rows


def choose_securities(portfolio_id: str) -> list[str]:
    """Return the realistic instrument universe assigned to one portfolio.

    The mapping is deterministic so injected exception IDs and evaluation labels
    remain stable across repeated data generation.
    """
    # Explicit holdings create realistic strategy composition and known injection
    # points. Random assignment would make ground-truth IDs unstable.
    mapping = {
        "MR-EQ-INDIA": [f"EQ{i:03d}" for i in range(1, 6)],
        "MR-EQ-GLOBAL": [f"EQ{i:03d}" for i in range(6, 12)],
        "MR-BOND-GOV": ["BD001", "BD002", "BD003", "BD004"],
        "MR-CREDIT-IG": ["BD005", "BD006", "BD007", "BD013"],
        "MR-CREDIT-HY": ["BD008", "BD009", "BD010", "BD012"],
        "MR-BALANCED": ["EQ006", "EQ007", "EQ013", "BD005", "BD006", "BD001"],
        "MR-INCOME-MC": ["BD002", "BD005", "BD007", "BD011"],
        "MR-ESG": ["EQ006", "EQ007", "EQ009", "EQ010", "EQ011", "EQ012"],
    }
    return mapping[portfolio_id]


def exception(
    exception_id: str,
    portfolio_id: str,
    category: str,
    severity: str,
    title: str,
    instrument_id: str | None,
    expected_agents: list[str],
    required_policies: list[str],
    approval_required: bool,
    expected_resolution: str,
    expected_materiality_bps: float = 0.0,
) -> dict[str, Any]:
    """Create one hidden expected-exception label.

    Args:
        exception_id: Stable ID expected from deterministic tools.
        portfolio_id: Portfolio where the condition was injected.
        category: Normalized exception family.
        severity: Expected risk classification.
        title: Human-readable condition name.
        instrument_id: Optional security involved in the issue.
        expected_agents: Specialists expected to participate.
        required_policy_ids: Policy evidence expected in a sound review.
        approval_required: Whether the scenario should reach HITL.
        expected_resolution: Target operational next step.
        expected_materiality_bps: Expected NAV impact for materiality tests.
    """
    return {
        "exception_id": exception_id,
        "portfolio_id": portfolio_id,
        "close_period": CLOSE_DATE.isoformat(),
        "category": category,
        "severity": severity,
        "title": title,
        "instrument_id": instrument_id,
        "expected_agents": expected_agents,
        "required_policy_ids": required_policies,
        "approval_required": approval_required,
        "expected_resolution": expected_resolution,
        "expected_materiality_bps": expected_materiality_bps,
    }


def generate_finance_data() -> list[dict[str, Any]]:
    """Generate clean populations, inject controlled breaks, and write labels.

    The function first creates coherent source records and then alters selected
    records differently in book, custodian, pricing, accounting, and mandate
    sources. That produces genuine cross-source problems rather than arbitrary
    rows labelled as errors.

    Returns:
        Ground-truth exceptions used only by the evaluation layer.
    """
    rng = random.Random(SEED)
    security_master = build_security_master()
    security_by_id = {row["security_id"]: row for row in security_master}
    portfolio_by_id = {row["portfolio_id"]: row for row in PORTFOLIOS}

    internal_positions: list[dict[str, Any]] = []
    for portfolio in PORTFOLIOS:
        selected = choose_securities(portfolio["portfolio_id"])
        target_invested = portfolio["nav"] * rng.uniform(0.82, 0.94)
        weights = [rng.uniform(0.7, 1.3) for _ in selected]
        if portfolio["portfolio_id"] == "MR-CREDIT-HY":
            weights[selected.index("BD009")] = 1.8
        if portfolio["portfolio_id"] == "MR-ESG":
            weights[selected.index("EQ006")] = 3.8
        weight_total = sum(weights)
        for security_id, weight in zip(selected, weights, strict=True):
            security = security_by_id[security_id]
            allocation = target_invested * weight / weight_total
            price = float(security["reference_price"])
            multiplier = 1000.0 if security["asset_class"] == "Bond" else 1.0
            quantity = max(1, round(allocation / (price * multiplier), 2))
            market_value = round(quantity * price * multiplier, 2)
            internal_positions.append({
                "close_period": CLOSE_DATE.isoformat(),
                "portfolio_id": portfolio["portfolio_id"],
                "security_id": security_id,
                "asset_class": security["asset_class"],
                "currency": security["currency"],
                "quantity": quantity,
                "price": price,
                "price_source": "VendorA",
                "market_value": market_value,
            })

    custodian_positions = [dict(row) for row in internal_positions]
    ground_truth: list[dict[str, Any]] = []

    # Position and custody exceptions.
    for portfolio_id, security_id, delta, severity in [
        ("MR-EQ-INDIA", "EQ003", 120.0, "high"),
        ("MR-EQ-GLOBAL", "EQ008", -75.0, "medium"),
        ("MR-CREDIT-IG", "BD006", 14.0, "high"),
    ]:
        row = next(r for r in custodian_positions if r["portfolio_id"] == portfolio_id and r["security_id"] == security_id)
        row["quantity"] = round(float(row["quantity"]) + delta, 2)
        row["market_value"] = round(float(row["quantity"]) * float(row["price"]) * (1000 if row["asset_class"] == "Bond" else 1), 2)
        ground_truth.append(exception(
            f"POS-{portfolio_id}-{security_id}", portfolio_id, "position_break", severity,
            f"Position quantity mismatch for {security_id}", security_id,
            ["reconciliation"], ["REC-001", "MAT-001"], severity == "high", "investigate_custody_or_settlement",
        ))

    for portfolio_id, security_id in [("MR-BOND-GOV", "BD004"), ("MR-INCOME-MC", "BD011")]:
        custodian_positions = [r for r in custodian_positions if not (r["portfolio_id"] == portfolio_id and r["security_id"] == security_id)]
        ground_truth.append(exception(
            f"POS-MISSING-{portfolio_id}-{security_id}", portfolio_id, "missing_custodian_position", "high",
            f"Internal position {security_id} missing from custodian record", security_id,
            ["reconciliation", "risk_policy"], ["REC-001", "MAT-001"], True, "confirm_settlement_or_correct_position",
        ))

    # Cash records and controlled differences.
    cash_rows: list[dict[str, Any]] = []
    custodian_cash: list[dict[str, Any]] = []
    for portfolio in PORTFOLIOS:
        currencies = sorted({security_by_id[s]["currency"] for s in choose_securities(portfolio["portfolio_id"])})
        if portfolio["base_currency"] not in currencies:
            currencies.append(portfolio["base_currency"])
        for currency in currencies:
            balance = round(portfolio["nav"] * rng.uniform(0.005, 0.025), 2)
            row = {"close_period": CLOSE_DATE.isoformat(), "portfolio_id": portfolio["portfolio_id"], "currency": currency, "balance": balance}
            cash_rows.append(row)
            custodian_cash.append(dict(row))

    for portfolio_id, currency, delta, severity in [
        ("MR-BALANCED", "USD", -250_000.0, "high"),
        ("MR-INCOME-MC", "EUR", 85_000.0, "medium"),
        ("MR-EQ-GLOBAL", "USD", 42_500.0, "medium"),
    ]:
        row = next(r for r in custodian_cash if r["portfolio_id"] == portfolio_id and r["currency"] == currency)
        row["balance"] = round(float(row["balance"]) + delta, 2)
        ground_truth.append(exception(
            f"CASH-{portfolio_id}-{currency}", portfolio_id, "cash_break", severity,
            f"Cash difference in {currency}", None,
            ["reconciliation", "accounting"], ["REC-001", "MAT-001"], severity == "high", "identify_timing_or_missing_cash_entry",
        ))

    # Pricing feeds: clean population followed by stale and conflict injections.
    vendor_a_prices: list[dict[str, Any]] = []
    vendor_b_prices: list[dict[str, Any]] = []
    for security in security_master:
        base_price = float(security["reference_price"])
        vendor_a_prices.append({
            "close_period": CLOSE_DATE.isoformat(), "security_id": security["security_id"],
            "price": round(base_price * rng.uniform(0.998, 1.002), 4), "currency": security["currency"],
            "as_of_date": CLOSE_DATE.isoformat(), "source": "VendorA", "quality": "official_close",
        })
        vendor_b_prices.append({
            "close_period": CLOSE_DATE.isoformat(), "security_id": security["security_id"],
            "price": round(base_price * rng.uniform(0.996, 1.004), 4), "currency": security["currency"],
            "as_of_date": CLOSE_DATE.isoformat(), "source": "VendorB", "quality": "independent_check",
        })

    for portfolio_id, security_id, stale_days, materiality in [
        ("MR-CREDIT-HY", "BD009", 5, 18.2),
        ("MR-BOND-GOV", "BD003", 4, 11.4),
        ("MR-INCOME-MC", "BD011", 3, 9.2),
    ]:
        row = next(r for r in vendor_a_prices if r["security_id"] == security_id)
        row["as_of_date"] = (CLOSE_DATE - timedelta(days=stale_days)).isoformat()
        row["quality"] = "stale"
        ground_truth.append(exception(
            f"VAL-STALE-{portfolio_id}-{security_id}", portfolio_id, "stale_price", "high" if materiality >= 10 else "medium",
            f"Price for {security_id} is {stale_days} days old", security_id,
            ["valuation", "risk_policy", "evidence_critic"], ["VAL-001", "MAT-001", "SRC-001"], materiality >= 10,
            "obtain_approved_alternative_price", materiality,
        ))

    for portfolio_id, security_id, pct_difference, materiality in [
        ("MR-CREDIT-HY", "BD010", -0.12, 24.8),
        ("MR-CREDIT-IG", "BD013", 0.045, 13.1),
        ("MR-BALANCED", "EQ013", -0.032, 7.5),
    ]:
        row = next(r for r in vendor_b_prices if r["security_id"] == security_id)
        row["price"] = round(float(row["price"]) * (1 + pct_difference), 4)
        row["quality"] = "price_conflict"
        ground_truth.append(exception(
            f"VAL-CONFLICT-{portfolio_id}-{security_id}", portfolio_id, "price_conflict", "high" if materiality >= 10 else "medium",
            f"Independent price sources disagree for {security_id}", security_id,
            ["valuation", "risk_policy", "evidence_critic"], ["VAL-001", "SRC-001", "MAT-001"], materiality >= 10,
            "price_challenge_and_human_review", materiality,
        ))

    # Transactions with two duplicates and realistic settlement states.
    transactions: list[dict[str, Any]] = []
    transaction_number = 1
    for position in internal_positions:
        if rng.random() > 0.58:
            continue
        trade_date = CLOSE_DATE - timedelta(days=rng.randint(1, 24))
        settlement_date = trade_date + timedelta(days=2)
        quantity = round(max(1.0, float(position["quantity"]) * rng.uniform(0.01, 0.08)), 2)
        transactions.append({
            "transaction_id": f"TX-{transaction_number:05d}",
            "portfolio_id": position["portfolio_id"],
            "security_id": position["security_id"],
            "transaction_type": rng.choice(["BUY", "SELL"]),
            "trade_date": trade_date.isoformat(),
            "settlement_date": settlement_date.isoformat(),
            "quantity": quantity,
            "price": position["price"],
            "currency": position["currency"],
            "status": "SETTLED" if settlement_date <= CLOSE_DATE else "PENDING",
        })
        transaction_number += 1

    for portfolio_id in ["MR-EQ-GLOBAL", "MR-BALANCED"]:
        original = next(r for r in transactions if r["portfolio_id"] == portfolio_id)
        duplicate = dict(original)
        duplicate["transaction_id"] = f"TX-{transaction_number:05d}"
        transaction_number += 1
        transactions.append(duplicate)
        ground_truth.append(exception(
            f"TX-DUP-{portfolio_id}-{original['security_id']}", portfolio_id, "duplicate_transaction", "high",
            f"Potential duplicate transaction for {original['security_id']}", original["security_id"],
            ["reconciliation", "accounting", "evidence_critic"], ["REC-001", "JRN-001"], True,
            "reverse_or_remove_duplicate_after_review",
        ))

    # Fees and accruals.
    fees: list[dict[str, Any]] = []
    fee_rates = {"management_fee": 0.0008, "administration_fee": 0.00012, "custody_fee": 0.00006, "audit_fee": 0.00003}
    for portfolio in PORTFOLIOS:
        for fee_type, rate in fee_rates.items():
            expected = round(portfolio["nav"] * rate, 2)
            fees.append({
                "close_period": CLOSE_DATE.isoformat(), "portfolio_id": portfolio["portfolio_id"],
                "fee_type": fee_type, "expected_accrual": expected, "booked_accrual": expected,
                "invoice_received": "true" if fee_type != "audit_fee" else "false", "currency": portfolio["base_currency"],
            })

    for portfolio_id, fee_type, multiplier, severity in [
        ("MR-BALANCED", "management_fee", 0.55, "high"),
        ("MR-ESG", "administration_fee", 0.0, "medium"),
        ("MR-CREDIT-IG", "custody_fee", 1.8, "medium"),
    ]:
        row = next(r for r in fees if r["portfolio_id"] == portfolio_id and r["fee_type"] == fee_type)
        row["booked_accrual"] = round(float(row["expected_accrual"]) * multiplier, 2)
        ground_truth.append(exception(
            f"FEE-{portfolio_id}-{fee_type}", portfolio_id, "fee_accrual_break", severity,
            f"Booked {fee_type} does not match expected accrual", None,
            ["accounting", "risk_policy", "evidence_critic"], ["FEE-001", "JRN-001", "MAT-001"], severity == "high",
            "prepare_adjusting_journal",
        ))

    # Corporate actions, including missing dividend/coupon records.
    corporate_actions = [
        {"action_id": "CA-001", "portfolio_id": "MR-EQ-INDIA", "security_id": "EQ005", "action_type": "DIVIDEND", "ex_date": "2026-10-19", "pay_date": "2026-10-30", "expected_amount": 425_000.0, "booked_amount": 0.0, "currency": "INR"},
        {"action_id": "CA-002", "portfolio_id": "MR-INCOME-MC", "security_id": "BD002", "action_type": "COUPON", "ex_date": "2026-10-01", "pay_date": "2026-10-28", "expected_amount": 180_000.0, "booked_amount": 142_000.0, "currency": "INR"},
        {"action_id": "CA-003", "portfolio_id": "MR-EQ-GLOBAL", "security_id": "EQ007", "action_type": "DIVIDEND", "ex_date": "2026-10-17", "pay_date": "2026-10-29", "expected_amount": 72_500.0, "booked_amount": 72_500.0, "currency": "USD"},
        {"action_id": "CA-004", "portfolio_id": "MR-CREDIT-IG", "security_id": "BD007", "action_type": "COUPON", "ex_date": "2026-10-05", "pay_date": "2026-10-25", "expected_amount": 96_000.0, "booked_amount": 96_000.0, "currency": "USD"},
    ]
    ground_truth.extend([
        exception("CA-MISSING-MR-EQ-INDIA-EQ005", "MR-EQ-INDIA", "missing_income", "high", "Dividend income was not booked", "EQ005", ["accounting", "reconciliation"], ["CA-001", "JRN-001"], True, "book_dividend_receivable"),
        exception("CA-COUPON-MR-INCOME-MC-BD002", "MR-INCOME-MC", "coupon_accrual_break", "medium", "Coupon income is under-accrued", "BD002", ["accounting", "evidence_critic"], ["CA-001", "JRN-001"], False, "correct_coupon_accrual"),
    ])

    # Ledger is built from internal data, then exceptions are injected.
    investment_totals: dict[str, float] = defaultdict(float)
    cash_totals: dict[str, float] = defaultdict(float)
    for row in internal_positions:
        investment_totals[row["portfolio_id"]] += float(row["market_value"])
    for row in cash_rows:
        cash_totals[row["portfolio_id"]] += float(row["balance"])

    ledger: list[dict[str, Any]] = []
    for portfolio in PORTFOLIOS:
        pid = portfolio["portfolio_id"]
        ledger.extend([
            {"close_period": CLOSE_DATE.isoformat(), "portfolio_id": pid, "account_code": "1100", "account_name": "Investments at fair value", "balance": round(investment_totals[pid], 2), "currency": portfolio["base_currency"]},
            {"close_period": CLOSE_DATE.isoformat(), "portfolio_id": pid, "account_code": "1200", "account_name": "Cash and cash equivalents", "balance": round(cash_totals[pid], 2), "currency": portfolio["base_currency"]},
            {"close_period": CLOSE_DATE.isoformat(), "portfolio_id": pid, "account_code": "2100", "account_name": "Accrued expenses", "balance": -round(sum(float(r["booked_accrual"]) for r in fees if r["portfolio_id"] == pid), 2), "currency": portfolio["base_currency"]},
        ])

    for portfolio_id, delta, severity in [
        ("MR-CREDIT-IG", 250_000.0, "high"),
        ("MR-BALANCED", -175_000.0, "high"),
    ]:
        row = next(r for r in ledger if r["portfolio_id"] == portfolio_id and r["account_code"] == "1100")
        row["balance"] = round(float(row["balance"]) + delta, 2)
        ground_truth.append(exception(
            f"GL-INV-{portfolio_id}", portfolio_id, "general_ledger_break", severity,
            "Investment subledger does not reconcile to the general ledger", None,
            ["reconciliation", "accounting", "evidence_critic"], ["REC-001", "JRN-001", "MAT-001"], True,
            "identify_and_approve_adjusting_journal",
        ))

    # Mandate breaches are based on position concentration and restricted flags.
    ground_truth.extend([
        exception("RISK-CONC-MR-ESG-EQ006", "MR-ESG", "concentration_breach", "high", "Single-name exposure exceeds the 15% ESG mandate limit", "EQ006", ["risk_policy", "evidence_critic"], ["RISK-001", "APP-001"], True, "escalate_to_risk_and_portfolio_manager", 22.7),
        exception("RISK-REST-MR-ESG-EQ012", "MR-ESG", "restricted_security", "critical", "Restricted thermal-coal issuer appears in ESG portfolio", "EQ012", ["risk_policy", "evidence_critic"], ["RISK-001", "APP-001"], True, "freeze_and_compliance_escalation"),
        exception("RISK-REST-MR-CREDIT-HY-BD012", "MR-CREDIT-HY", "restricted_security", "critical", "Restricted issuer appears in opportunistic credit portfolio", "BD012", ["risk_policy", "evidence_critic"], ["RISK-001", "APP-001"], True, "freeze_and_compliance_escalation"),
    ])

    # Portfolio metadata and source manifests.
    portfolio_rows = [
        {
            **portfolio,
            "close_period": CLOSE_DATE.isoformat(),
            "materiality_review_bps": 5.0,
            "materiality_approval_bps": 10.0,
        }
        for portfolio in PORTFOLIOS
    ]
    manifest = [
        {"source_name": name, "file_name": filename, "expected_frequency": "monthly", "authoritative_for": authority, "close_period": CLOSE_DATE.isoformat()}
        for name, filename, authority in [
            ("Portfolio Accounting", "internal_positions.csv", "internal positions and book values"),
            ("Global Custodian", "custodian_positions.csv", "external positions"),
            ("Treasury Ledger", "internal_cash.csv", "internal cash"),
            ("Global Custodian", "custodian_cash.csv", "external cash"),
            ("Vendor A", "vendor_a_prices.csv", "primary closing prices"),
            ("Vendor B", "vendor_b_prices.csv", "independent price challenge"),
            ("General Ledger", "general_ledger.csv", "official accounting balances"),
            ("Expense Platform", "fee_accruals.csv", "fees and accruals"),
            ("Trading Platform", "transactions.csv", "executed trades"),
            ("Asset Servicing", "corporate_actions.csv", "income and corporate actions"),
        ]
    ]

    write_csv(GENERATED / "security_master.csv", security_master)
    write_csv(GENERATED / "portfolios.csv", portfolio_rows)
    write_csv(GENERATED / "internal_positions.csv", internal_positions)
    write_csv(GENERATED / "custodian_positions.csv", custodian_positions)
    write_csv(GENERATED / "internal_cash.csv", cash_rows)
    write_csv(GENERATED / "custodian_cash.csv", custodian_cash)
    write_csv(GENERATED / "vendor_a_prices.csv", vendor_a_prices)
    write_csv(GENERATED / "vendor_b_prices.csv", vendor_b_prices)
    write_csv(GENERATED / "transactions.csv", transactions)
    write_csv(GENERATED / "fee_accruals.csv", fees)
    write_csv(GENERATED / "corporate_actions.csv", corporate_actions)
    write_csv(GENERATED / "general_ledger.csv", ledger)
    write_csv(GENERATED / "source_manifest.csv", manifest)
    write_json(GROUND_TRUTH / "exceptions.json", ground_truth)

    scenarios = [
        {"scenario_id": "multi_issue_close", "name": "Enterprise month-end close", "description": "All eight portfolios with pricing, reconciliation, accounting, policy, and evidence exceptions.", "portfolio_id": None, "expected_focus": ["cross-portfolio prioritization", "parallel specialists", "multiple approvals"]},
        {"scenario_id": "material_pricing_case", "name": "Material high-yield pricing case", "description": "Stale and conflicting prices in the opportunistic credit portfolio.", "portfolio_id": "MR-CREDIT-HY", "expected_focus": ["A2A valuation agent", "materiality", "human approval"]},
        {"scenario_id": "accounting_close_pressure", "name": "Balanced fund accounting pressure", "description": "Cash, fee, price, and ledger exceptions near the close deadline.", "portfolio_id": "MR-BALANCED", "expected_focus": ["accounting", "reconciliation", "approval"]},
        {"scenario_id": "esg_control_breach", "name": "ESG mandate breach", "description": "Concentration and restricted-security issues requiring risk escalation.", "portfolio_id": "MR-ESG", "expected_focus": ["policy RAG", "risk specialist", "zero-tolerance controls"]},
        {"scenario_id": "multi_currency_income", "name": "Multi-currency income review", "description": "FX cash, coupon, custody, and stale-price issues.", "portfolio_id": "MR-INCOME-MC", "expected_focus": ["income accrual", "currency cash", "evidence completeness"]},
    ]
    write_json(SCENARIOS / "scenarios.json", scenarios)
    write_json(GENERATED / "dataset_profile.json", {
        "seed": SEED,
        "close_period": CLOSE_DATE.isoformat(),
        "portfolio_count": len(PORTFOLIOS),
        "portfolio_ids": [item["portfolio_id"] for item in PORTFOLIOS],
        "security_count": len(security_master),
        "internal_position_rows": len(internal_positions),
        "custodian_position_rows": len(custodian_positions),
        "transaction_rows": len(transactions),
        "injected_exception_count": len(ground_truth),
        "purpose": "Synthetic month-end close data for agentic workflow and evaluation. Not investment advice.",
    })
    return ground_truth


POLICY_DOCUMENTS: dict[str, str] = {
    "REC-001_month_end_reconciliation.md": """# Month-End Reconciliation and Close Control Policy

**Policy ID:** REC-001  
**Version:** 3.2  
**Effective date:** 2026-07-01  
**Owner:** Head of Portfolio Accounting  
**Applies to:** All Meridian Ridge portfolios

## Purpose

Every reporting portfolio must have a complete and independently supported population of positions, cash, transactions, income, expenses, and ledger balances before final close approval. Completion of a system job is not evidence that the population is complete.

## Mandatory source checks

The close owner must verify receipt of the internal accounting population, custodian positions, custodian cash, executed trades, primary prices, corporate actions, fee records, and general-ledger balances. Missing mandatory sources block final approval unless the Finance Controller grants a documented exception.

## Reconciliation rules

Position, cash, transaction, and ledger reconciliations must be performed at portfolio, currency, instrument, and reporting-period level. Netting unrelated breaks is prohibited. Each break must retain its gross difference, source values, age, owner, status, and supporting evidence.

Timing differences may be accepted only when an authoritative settlement or cash record identifies the expected clearing date. A verbal explanation is insufficient. Breaks open for more than two business days must be escalated to the Portfolio Accounting Manager.

## Close blocking conditions

The close must stop when a mandatory source is missing, the population control total fails, a material break lacks an owner, or a critical compliance exception remains unresolved. System availability does not override these conditions.

## Evidence

Required evidence includes source-file identifiers, reporting period, extraction timestamp, reconciliation calculation, reviewer, resolution, and approval where applicable. The final package must retain unresolved items separately from resolved items.
""",
    "SRC-001_source_authority.md": """# Financial Source Authority and Evidence Hierarchy

**Policy ID:** SRC-001  
**Version:** 2.4  
**Effective date:** 2026-06-15  
**Owner:** Finance Data Governance Council

## Authority hierarchy

Approved books and official custodian records are the highest-authority records for accounting balances and externally held assets. Approved pricing vendors are authoritative only for the instrument types and valuation times listed in the pricing matrix. Policies and mandates determine interpretation; emails and analyst notes provide context but never replace an approved policy.

## Conflict handling

Evidence must match the portfolio, legal entity, security, currency, and effective period. When approved sources disagree, the investigator must apply the documented source hierarchy, assess effective dates and quality flags, and retain both values. Averaging conflicting values without an approved methodology is prohibited.

If authority remains ambiguous, the case must be classified as unresolved and escalated. Retrieval similarity or model confidence cannot make a source authoritative.

## Minimum evidence metadata

Every supported claim must retain source ID, source type, owner, effective date, version, portfolio scope, transformation history, and citation location. Evidence missing required metadata may be presented as an investigative lead but not as a final basis for approval.
""",
    "VAL-001_pricing_and_fair_value.md": """# Pricing, Stale Price, and Fair-Value Review Policy

**Policy ID:** VAL-001  
**Version:** 4.1  
**Effective date:** 2026-08-01  
**Owner:** Valuation Committee

## Primary pricing

Listed equities use the approved market close from Vendor A. Government and investment-grade bonds use Vendor A evaluated prices with Vendor B as an independent challenge source. High-yield, distressed, and illiquid instruments require additional review when sources differ materially.

## Stale prices

An equity price is stale when it is older than one active market day. A bond price is stale when it is older than two calendar days unless the instrument is documented as non-trading. Stale prices must not be silently carried forward.

## Price challenges

A price difference above 2% for investment-grade instruments or 3% for high-yield instruments requires investigation. The reviewer must consider market movement, security terms, liquidity, trade evidence, broker quotes, source quality, and prior approved challenges.

## Fair-value adjustments

The valuation specialist may recommend a fair-value adjustment but may not approve a material override. A recommendation must include original prices, alternative evidence, calculation, NAV impact, uncertainty, and residual risk.

Overrides producing an impact of 10 basis points or more require Finance Controller and Valuation Committee approval. Restricted or sanctioned instruments require Compliance review regardless of amount.
""",
    "MAT-001_materiality_and_escalation.md": """# Materiality and Escalation Standard

**Policy ID:** MAT-001  
**Version:** 2.8  
**Effective date:** 2026-07-01  
**Owner:** Finance Risk Committee

## Materiality dimensions

Materiality is not determined by amount alone. Reviewers must consider basis-point impact on NAV, client impact, regulatory exposure, mandate breach, uncertainty, repetition, fraud indicators, and proximity to the reporting deadline.

## Thresholds

Items below 5 basis points may follow standard reviewer approval when evidence is complete and no policy breach exists. Items from 5 to below 10 basis points require documented specialist review. Items of 10 basis points or more require Finance Controller approval. Critical compliance and restricted-security cases require escalation regardless of basis-point impact.

## Confidence and evidence

Low model confidence, conflicting authoritative sources, missing mandatory evidence, or repeated failed investigation attempts require human escalation. A high-confidence model output does not reduce the approval requirement.

## Aggregation

Related breaks must be considered both individually and in aggregate. Unrelated breaks must not be netted to fall below a threshold.
""",
    "JRN-001_journal_approval.md": """# Journal Preparation and Approval Policy

**Policy ID:** JRN-001  
**Version:** 3.0  
**Effective date:** 2026-05-01  
**Owner:** Corporate Controller

## Preparation

Adjusting journals must identify portfolio, legal entity, account, currency, amount, close period, reason, source evidence, and preparer. Automated systems may calculate and draft journal proposals but may not post journals using reviewer credentials.

## Segregation of duties

The preparer and approver must be different identities. Journals affecting valuation, fees, or income above the applicable materiality threshold require specialist review before controller approval.

## Duplicate prevention

Every proposed journal requires an idempotency key derived from portfolio, period, source exception, and journal type. A matching posted or pending journal must block a duplicate submission.

## Evidence and retention

The approved journal package must include calculation, policy basis, supporting source, comments, approval timestamp, and posting reference. Rejected proposals remain in the audit history.
""",
    "FEE-001_fees_and_accruals.md": """# Fees, Expenses, and Accruals Policy

**Policy ID:** FEE-001  
**Version:** 2.1  
**Effective date:** 2026-04-01  
**Owner:** Fund Expense Oversight

## Monthly accruals

Management, administration, custody, audit, legal, and other recurring expenses must be accrued using the approved fee schedule and current NAV or contractual basis. Missing invoices do not automatically remove the obligation to accrue an estimable expense.

## Variance review

Booked accruals must be compared with independently calculated expectations. Variances above 10% or above the portfolio materiality threshold require investigation. Zero accruals for active recurring services are presumed incomplete until supported.

## Allocation

Shared expenses must be allocated using the approved driver. Manual allocations require evidence of the driver and approval by Fund Finance. Expense reallocations between portfolios must never be performed solely to reduce a material exception.

## Adjustment

The accounting specialist may prepare a proposed adjustment. Posting requires a separate approver under JRN-001.
""",
    "CA-001_corporate_actions.md": """# Corporate Actions and Investment Income Policy

**Policy ID:** CA-001  
**Version:** 2.6  
**Effective date:** 2026-03-15  
**Owner:** Asset Servicing Operations

## Scope

Dividends, coupons, redemptions, splits, mergers, tax reclaims, and other entitlements must be recorded in the correct portfolio, security, currency, and period.

## Recognition

Dividend and coupon income must be recognized according to the approved accounting basis and supported by an authoritative announcement or custodian entitlement. Expected amounts must be compared with booked income and receivables.

## Exceptions

Missing or partial income is investigated using security terms, position entitlement, ex-date, record date, pay date, withholding tax, and custodian status. Material missing income requires accounting review and may require an adjusting journal.

## Evidence

The case must retain the event identifier, affected position, expected amount, booked amount, calculation, currency, source announcement, and final disposition.
""",
    "RISK-001_portfolio_mandates.md": """# Portfolio Mandate, Concentration, and Restricted Holdings Policy

**Policy ID:** RISK-001  
**Version:** 5.0  
**Effective date:** 2026-09-01  
**Owner:** Chief Risk and Compliance Officer

## Mandate checks

Every portfolio must be evaluated against its approved concentration, asset-class, issuer, rating, currency, and restricted-security rules using the final month-end population.

## Concentration

Single-name exposure is calculated using gross market value divided by portfolio NAV. The limit in the portfolio master is binding. Breaches may not be masked by netting derivatives or expected future trades unless the mandate explicitly permits that treatment.

## Restricted holdings

Restricted and sanctioned securities trigger immediate critical escalation regardless of size. The system must freeze automated recommendations that would increase or normalize the exposure. Compliance determines disposition.

## Human authority

Agents may identify and explain a breach. Only authorized Risk, Compliance, and portfolio governance personnel may accept, waive, or remediate it.
""",
    "APP-001_approval_matrix.md": """# Month-End Approval Authority Matrix

**Policy ID:** APP-001  
**Version:** 3.5  
**Effective date:** 2026-07-01  
**Owner:** Chief Financial Officer

## Decision rights

Portfolio Accounting owns reconciliation preparation. Valuation owns price-challenge recommendations. Fund Finance owns fee and accounting review. Risk and Compliance own mandate interpretation. The Finance Controller challenges close readiness. The Fund Finance Head provides final close approval.

## Approval levels

Routine evidence-complete items below 5 basis points may be approved by the assigned manager. Items from 5 to below 10 basis points require a domain specialist and portfolio accounting manager. Items at or above 10 basis points require Finance Controller approval. Policy waivers, restricted holdings, or client-reporting impact require the designated senior authority regardless of size.

## Prohibited delegation

No AI agent, service account, workflow engine, or document may grant itself approval authority. Approval must be associated with an authenticated human identity, authorized role, timestamp, decision, and comments.

## Returned work

Approvers may request more information. The case must preserve the prior recommendation and record the additional work instead of overwriting history.
""",
    "AUD-001_audit_and_retention.md": """# Audit Evidence, Traceability, and Retention Policy

**Policy ID:** AUD-001  
**Version:** 2.3  
**Effective date:** 2026-01-01  
**Owner:** Internal Control Assurance

## Required trace

The close record must identify source files, versions, transformations, calculations, agent and tool actions, human decisions, model and prompt versions where applicable, errors, retries, and final outputs.

## Reproducibility

A reviewer must be able to reproduce material calculations using retained inputs. Generated narrative without underlying calculation and source evidence is insufficient.

## Agent records

Agent logs must contain structured actions and outputs but must not depend on private chain-of-thought. Task IDs, timestamps, selected skills, tool calls, evidence references, and decision summaries are required.

## Retention and access

Access follows least privilege and portfolio scope. Evidence must not be copied across portfolios without authorization. Rejected, failed, and canceled tasks remain visible in the audit history.
""",
}


def generate_policies() -> None:
    """Write detailed versioned control policies and a searchable catalog."""
    POLICIES.mkdir(parents=True, exist_ok=True)
    for filename, content in POLICY_DOCUMENTS.items():
        (POLICIES / filename).write_text(content.strip() + "\n", encoding="utf-8")
    write_json(POLICIES / "policy_catalog.json", [
        {
            "policy_id": filename.split("_")[0],
            "file_name": filename,
            "title": content.splitlines()[0].removeprefix("# "),
            "effective_date": next(line.split(":**", 1)[-1].strip() for line in content.splitlines() if line.startswith("**Effective date:")),
        }
        for filename, content in POLICY_DOCUMENTS.items()
    ])


def main() -> None:
    """Idempotent command-line entry point for rebuilding all synthetic assets."""
    for directory in [GENERATED, POLICIES, GROUND_TRUTH, SCENARIOS]:
        directory.mkdir(parents=True, exist_ok=True)
    ground_truth = generate_finance_data()
    generate_policies()
    print(json.dumps({
        "status": "generated",
        "root": str(DATA),
        "portfolios": len(PORTFOLIOS),
        "policies": len(POLICY_DOCUMENTS),
        "ground_truth_exceptions": len(ground_truth),
    }, indent=2))


if __name__ == "__main__":
    main()

