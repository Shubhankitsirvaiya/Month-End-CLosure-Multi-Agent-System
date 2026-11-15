"""Load the generated month-end population as a coherent data bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.core.config import Settings, get_settings


@dataclass(slots=True)
class CloseDataBundle:
    """All independent source-system populations needed for one close.

    A dataclass gives readable attribute access while preserving each source as
    its own DataFrame. Combining sources too early would hide reconciliation
    differences—the very conditions this project is designed to detect.
    """
    portfolios: pd.DataFrame
    security_master: pd.DataFrame
    internal_positions: pd.DataFrame
    custodian_positions: pd.DataFrame
    internal_cash: pd.DataFrame
    custodian_cash: pd.DataFrame
    vendor_a_prices: pd.DataFrame
    vendor_b_prices: pd.DataFrame
    transactions: pd.DataFrame
    fee_accruals: pd.DataFrame
    corporate_actions: pd.DataFrame
    general_ledger: pd.DataFrame
    source_manifest: pd.DataFrame

    def for_portfolio(self, portfolio_id: str | None) -> "CloseDataBundle":
        """Return a portfolio-scoped copy while preserving global reference data.

        Args:
            portfolio_id: Portfolio identifier to keep. ``None`` intentionally
                means enterprise-wide scope and returns the current bundle.

        Returns:
            A new bundle. Frames containing ``portfolio_id`` are filtered;
            security master and market-price feeds remain global because they
            are reference datasets shared across portfolios.
        """

        # None/empty means an enterprise-wide close; do not filter any frame.
        if not portfolio_id:
            return self

        def filter_frame(frame: pd.DataFrame) -> pd.DataFrame:
            """Filter portfolio-owned rows while copying global reference frames."""
            # Prices and security master are shared reference populations.
            if "portfolio_id" not in frame.columns:
                return frame.copy()
            # .copy() prevents later agent code from mutating source data.
            return frame.loc[frame["portfolio_id"] == portfolio_id].copy()

        return CloseDataBundle(
            portfolios=filter_frame(self.portfolios),
            security_master=self.security_master.copy(),
            internal_positions=filter_frame(self.internal_positions),
            custodian_positions=filter_frame(self.custodian_positions),
            internal_cash=filter_frame(self.internal_cash),
            custodian_cash=filter_frame(self.custodian_cash),
            vendor_a_prices=self.vendor_a_prices.copy(),
            vendor_b_prices=self.vendor_b_prices.copy(),
            transactions=filter_frame(self.transactions),
            fee_accruals=filter_frame(self.fee_accruals),
            corporate_actions=filter_frame(self.corporate_actions),
            general_ledger=filter_frame(self.general_ledger),
            source_manifest=self.source_manifest.copy(),
        )


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    """Read one required CSV and raise an actionable setup error if absent.

    Args:
        root: Directory containing generated source-system files.
        name: Exact CSV filename, including its extension.
    """
    # pathlib joins paths safely on Windows and Linux containers.
    path = root / name
    if not path.exists():
        raise FileNotFoundError(
            f"Required generated dataset is missing: {path}. "
            "Run `python -m scripts.generate_demo_data`."
        )
    # Business-specific numeric/date conversions happen inside controlled tools.
    return pd.read_csv(path)


def load_close_data(settings: Settings | None = None) -> CloseDataBundle:
    """Load every finance population into one explicitly named bundle."""
    settings = settings or get_settings()
    # Every filename is explicit so a missing source cannot be silently skipped.
    root = settings.data_root / "generated"
    return CloseDataBundle(
        portfolios=_read_csv(root, "portfolios.csv"),
        security_master=_read_csv(root, "security_master.csv"),
        internal_positions=_read_csv(root, "internal_positions.csv"),
        custodian_positions=_read_csv(root, "custodian_positions.csv"),
        internal_cash=_read_csv(root, "internal_cash.csv"),
        custodian_cash=_read_csv(root, "custodian_cash.csv"),
        vendor_a_prices=_read_csv(root, "vendor_a_prices.csv"),
        vendor_b_prices=_read_csv(root, "vendor_b_prices.csv"),
        transactions=_read_csv(root, "transactions.csv"),
        fee_accruals=_read_csv(root, "fee_accruals.csv"),
        corporate_actions=_read_csv(root, "corporate_actions.csv"),
        general_ledger=_read_csv(root, "general_ledger.csv"),
        source_manifest=_read_csv(root, "source_manifest.csv"),
    )


def load_scenarios(settings: Settings | None = None) -> list[dict]:
    """Load curated demonstration scenarios shown in the Streamlit selector."""
    settings = settings or get_settings()
    path = settings.data_root / "scenarios" / "scenarios.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario catalog missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_profile(settings: Settings | None = None) -> dict:
    """Load row counts and portfolio IDs used for UI context and validation."""
    settings = settings or get_settings()
    path = settings.data_root / "generated" / "dataset_profile.json"
    return json.loads(path.read_text(encoding="utf-8"))

