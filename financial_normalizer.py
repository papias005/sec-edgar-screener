"""
financial_normalizer.py
-----------------------
Parses the raw SEC XBRL Company Facts JSON produced by DataExtractor and
converts it into a clean, multi-index pandas DataFrame indexed by
(Year, Quarter) for downstream analysis and screening.

Phase 2 expansion: TAG_MAP now covers every US-GAAP XBRL tag required to
compute all 8 composite metrics in MetricsCalculator.

Key design rules:
  • Multiple XBRL tags can map to the same column name — they are merged
    via combine_first() so the most-reported tag wins.
  • Duration-based deduplication ensures we always prefer the longest
    reporting period (avoids picking restated/amended duplicates).
  • Missing tags produce NaN — never imputed.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US-GAAP XBRL tag → clean column name mapping (Phase 2 expanded).
# Tags that map to the same column are merged with combine_first().
# Order matters within same-target-column groups (first = preferred).
# ---------------------------------------------------------------------------
TAG_MAP: dict[str, str] = {
    # ── Income Statement ────────────────────────────────────────────────────
    "NetIncomeLoss": "net_income",
    "NetIncomeLossAvailableToCommonStockholdersBasic": "net_income",

    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",

    "OperatingIncomeLoss": "operating_income",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": "ebit",

    "InterestExpense": "interest_expense",
    "InterestAndDebtExpense": "interest_expense",

    "IncomeTaxExpenseBenefit": "income_tax_expense",

    # EPS / shares
    "EarningsPerShareBasic": "eps_basic",
    "EarningsPerShareDiluted": "eps_diluted",

    # ── Cash Flow Statement ─────────────────────────────────────────────────
    "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",

    "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
    "PaymentsForCapitalImprovements": "capex",

    "DepreciationDepletionAndAmortization": "da",
    "DepreciationAndAmortization": "da",
    "Depreciation": "da",

    # ── Balance Sheet — Assets ──────────────────────────────────────────────
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",

    "CashAndCashEquivalentsAtCarryingValue": "cash",
    "CashCashEquivalentsAndShortTermInvestments": "cash",
    "CashAndCashEquivalentsPeriodIncreaseDecrease": "cash",  # fallback

    "AccountsReceivableNetCurrent": "accounts_receivable",
    "ReceivablesNetCurrent": "accounts_receivable",

    "InventoryNet": "inventory",

    # ── Balance Sheet — Liabilities ─────────────────────────────────────────
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",

    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "LongTermDebtAndCapitalLeaseObligations": "long_term_debt",

    "ShortTermBorrowings": "short_term_debt",
    "DebtCurrent": "short_term_debt",
    "NotesPayableCurrent": "short_term_debt",

    "AccountsPayableCurrent": "accounts_payable",

    # ── Balance Sheet — Equity ──────────────────────────────────────────────
    "StockholdersEquity": "equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "equity",

    "RetainedEarningsAccumulatedDeficit": "retained_earnings",

    # ── Share data ──────────────────────────────────────────────────────────
    "CommonStockSharesOutstanding": "shares_outstanding",
    "EntityCommonStockSharesOutstanding": "shares_outstanding",
    "CommonStockSharesIssued": "shares_outstanding",  # fallback
}

# We only care about annual (10-K) and quarterly (10-Q) forms.
ACCEPTED_FORMS: frozenset[str] = frozenset({"10-K", "10-Q", "10-K/A", "10-Q/A"})


def _period_to_year_quarter(end_date: str, form: str) -> tuple[int, int | str]:
    """
    Derive (Year, Quarter) from an SEC filing end-date string (YYYY-MM-DD).

    Annual filings (10-K) get Quarter = 'A' (annual).
    Quarterly filings (10-Q) get Quarter = 1-4.
    """
    parts = end_date.split("-")
    year = int(parts[0])
    month = int(parts[1])
    if "10-K" in form:
        return year, "A"
    quarter = (month - 1) // 3 + 1
    return year, quarter


class FinancialNormalizer:
    """
    Converts the raw XBRL Company Facts JSON into a clean pandas DataFrame.

    The resulting DataFrame has a MultiIndex of (year, quarter) and one
    column per financial metric defined in TAG_MAP.

    Parameters
    ----------
    tag_map : dict[str, str] | None
        Override or extend the default TAG_MAP.  Keys are US-GAAP XBRL tags,
        values are clean column names.
    accepted_forms : frozenset[str] | None
        SEC form types to include.  Defaults to ACCEPTED_FORMS.
    prefer_usd : bool
        If True (default), only extract values reported in USD.

    Examples
    --------
    >>> normalizer = FinancialNormalizer()
    >>> df = normalizer.normalize(raw_json, ticker="AAPL")
    >>> print(df.head())
    """

    def __init__(
        self,
        tag_map: dict[str, str] | None = None,
        accepted_forms: frozenset[str] | None = None,
        prefer_usd: bool = True,
    ):
        self._tag_map = tag_map if tag_map is not None else TAG_MAP
        self._accepted_forms = (
            accepted_forms if accepted_forms is not None else ACCEPTED_FORMS
        )
        self._prefer_usd = prefer_usd

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_tag_series(
        self,
        facts_section: dict[str, Any],
        xbrl_tag: str,
    ) -> dict[tuple[int, int | str], float]:
        """
        Pull all values for a single XBRL tag from the 'us-gaap' namespace.

        Returns a dict keyed by (Year, Quarter) → value (float).
        When multiple entries exist for the same period, the one with the
        longest reporting duration is kept (12-month >> restated amendment).
        """
        us_gaap = facts_section.get("us-gaap", {})
        # Also check DEI namespace for share-count tags
        dei = facts_section.get("dei", {})

        tag_data = us_gaap.get(xbrl_tag) or dei.get(xbrl_tag, {})
        units_dict = tag_data.get("units", {})

        if not units_dict:
            return {}

        # Prefer USD; for share counts prefer 'shares'; otherwise first unit
        if self._prefer_usd and "USD" in units_dict:
            entries = units_dict["USD"]
        elif "shares" in units_dict:
            entries = units_dict["shares"]
        else:
            entries = next(iter(units_dict.values()))

        period_values: dict[tuple[int, int | str], tuple[float, int]] = {}

        for entry in entries:
            form = entry.get("form", "")
            if form not in self._accepted_forms:
                continue

            end_date = entry.get("end")
            val = entry.get("val")
            if end_date is None or val is None:
                continue

            try:
                key = _period_to_year_quarter(end_date, form)
            except (ValueError, AttributeError, IndexError):
                continue

            # Compute duration (days) to prefer full-year over quarter restates
            start = entry.get("start")
            duration = 0
            if start:
                try:
                    duration = (pd.Timestamp(end_date) - pd.Timestamp(start)).days
                except Exception:
                    duration = 0

            existing = period_values.get(key)
            if existing is None or duration > existing[1]:
                period_values[key] = (float(val), duration)

        return {k: v[0] for k, v in period_values.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, raw_json: dict, ticker: str = "") -> pd.DataFrame:
        """
        Parse the raw XBRL Company Facts JSON and return a tidy DataFrame.

        Parameters
        ----------
        raw_json : dict
            The full Company Facts JSON as returned by DataExtractor.
        ticker : str
            Optional ticker label used only for logging.

        Returns
        -------
        pd.DataFrame
            MultiIndex (year, quarter) DataFrame with one column per metric.
            Quarter is either an integer (1-4) or the string 'A' for annual.
            Unknown / missing periods produce NaN.
        """
        label = ticker or raw_json.get("entityName", "UNKNOWN")
        logger.info("FinancialNormalizer: normalising data for %s", label)

        facts = raw_json.get("facts", {})
        if not facts:
            logger.warning(
                "FinancialNormalizer: no 'facts' found in payload for %s", label
            )
            return pd.DataFrame()

        series_dict: dict[str, pd.Series] = {}

        for xbrl_tag, col_name in self._tag_map.items():
            period_values = self._extract_tag_series(facts, xbrl_tag)
            if not period_values:
                logger.debug(
                    "FinancialNormalizer: tag '%s' not found for %s", xbrl_tag, label
                )
                continue

            series = pd.Series(period_values, name=col_name)
            series.index = pd.MultiIndex.from_tuples(
                series.index, names=["year", "quarter"]
            )

            # Merge duplicate-target-column tags via combine_first
            # (earlier entries in TAG_MAP take priority)
            if col_name in series_dict:
                series_dict[col_name] = series_dict[col_name].combine_first(series)
            else:
                series_dict[col_name] = series

            logger.debug(
                "FinancialNormalizer: extracted %d rows for tag '%s' (%s) on %s",
                len(period_values),
                xbrl_tag,
                col_name,
                label,
            )

        if not series_dict:
            logger.warning(
                "FinancialNormalizer: no relevant tags found for %s", label
            )
            return pd.DataFrame()

        df = pd.DataFrame(series_dict)
        df.sort_index(inplace=True)

        logger.info(
            "FinancialNormalizer: %s → DataFrame shape %s, columns: %s",
            label,
            df.shape,
            list(df.columns),
        )
        return df
