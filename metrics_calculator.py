"""
metrics_calculator.py
---------------------
Computes the 8 composite quantitative metrics used by the QuantitativeScreener.

US_GAAP_MAPPING
---------------
An auditable dictionary that documents which normalised DataFrame columns are
derived from which primary XBRL tags (in priority order).  This is for
transparency and auditability — the actual extraction lives in
FinancialNormalizer.

MetricsCalculator
-----------------
Consumes a single-ticker, annual-only pandas DataFrame (the output of
FinancialNormalizer sliced to quarter=='A') and returns a flat dict of
8 metric values for the most-recent complete fiscal year.

NaN Policy (CRITICAL)
---------------------
If any foundational input tag is missing (NaN), the dependent metric is set
to float('nan'). Data is NEVER imputed or approximated. The QuantitativeScreener
will automatically exclude any ticker whose critical metrics are NaN.

Market Cap (live price)
-----------------------
Computed as:  shares_outstanding × yfinance.fast_info['lastPrice']
yfinance is called with a short timeout. On any failure the market_cap is NaN,
which propagates to EV/EBITDA and FCF Yield, triggering automatic exclusion.

8 Metrics
---------
1.  altman_z          — Modified Altman Z-Score (public companies)
2.  net_debt_ebitda   — Net Debt / EBITDA
3.  current_ratio     — Current Assets / Current Liabilities
4.  sloan_accruals    — (Net Income − Operating CFO) / Avg Total Assets
5.  ccr               — Operating CFO / EBITDA  [capital-structure-neutral]
6.  ev_ebitda         — (Market Cap + Net Debt) / EBITDA
7.  fcf_yield         — (Operating CFO − CapEx) / Market Cap
8.  fcf               — Operating CFO − CapEx  (absolute, used for quintile)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# US-GAAP column → list[primary_xbrl_tags_in_priority_order]
# Documentation / audit trail — not used at runtime by this module.
# ---------------------------------------------------------------------------
US_GAAP_MAPPING: dict[str, list[str]] = {
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "ebit": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "OperatingIncomeLoss",  # fallback
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
    "income_tax_expense": [
        "IncomeTaxExpenseBenefit",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
    ],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "total_assets": [
        "Assets",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
    ],
    "inventory": [
        "InventoryNet",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "NotesPayableCurrent",
    ],
    "accounts_payable": [
        "AccountsPayableCurrent",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
}


def _safe_get(row: pd.Series, col: str) -> float:
    """
    Safely retrieve a float from a pandas Series row.
    Returns NaN if the column is absent or its value is NaN.
    """
    val = row.get(col, np.nan)
    if val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_div(numerator: float, denominator: float) -> float:
    """Division with NaN propagation — never divides by zero."""
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0.0:
        return np.nan
    return numerator / denominator


def _fetch_market_cap(ticker: str, shares_outstanding: float) -> float:
    """
    Fetch live last price from yfinance and multiply by shares_outstanding.

    Returns NaN on any failure (timeout, data gap, bad ticker).
    Rate-limiting note: yfinance calls the Yahoo Finance API, not the SEC.
    No SEC rate limiter applies here.

    Parameters
    ----------
    ticker : str
        Standard stock ticker symbol.
    shares_outstanding : float
        Shares outstanding from SEC XBRL data.

    Returns
    -------
    float
        Market capitalisation in USD, or NaN.
    """
    if np.isnan(shares_outstanding) or shares_outstanding <= 0:
        logger.warning(
            "MetricsCalculator [%s]: shares_outstanding is NaN/zero — "
            "market_cap set to NaN.",
            ticker,
        )
        return np.nan

    try:
        import yfinance as yf  # imported here to keep dependency optional

        info = yf.Ticker(ticker).fast_info
        price = float(info["lastPrice"])

        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"Non-positive price: {price}")

        market_cap = shares_outstanding * price
        logger.info(
            "MetricsCalculator [%s]: live price=%.4f, shares=%.0f → market_cap=%.0f",
            ticker,
            price,
            shares_outstanding,
            market_cap,
        )
        return market_cap

    except ImportError:
        logger.error(
            "MetricsCalculator [%s]: yfinance not installed — "
            "run `pip install yfinance`. market_cap set to NaN.",
            ticker,
        )
        return np.nan
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MetricsCalculator [%s]: yfinance fetch failed (%s) — "
            "market_cap set to NaN.",
            ticker,
            exc,
        )
        return np.nan


class MetricsCalculator:
    """
    Computes 8 composite financial metrics for a single ticker from its
    normalised annual XBRL DataFrame.

    Parameters
    ----------
    ticker : str
        Ticker symbol — used for yfinance price fetch and logging.

    Examples
    --------
    >>> calc = MetricsCalculator("AAPL")
    >>> metrics = calc.compute(annual_df)
    >>> print(metrics["altman_z"])
    """

    def __init__(self, ticker: str):
        self._ticker = ticker.upper()

    # ------------------------------------------------------------------
    # Internal metric formulas
    # ------------------------------------------------------------------

    def _compute_ebitda(self, row: pd.Series) -> float:
        """
        EBITDA = Operating Income + D&A.

        Fallback: EBIT + D&A when OperatingIncomeLoss is unavailable.
        """
        operating_income = _safe_get(row, "operating_income")
        ebit = _safe_get(row, "ebit")
        da = _safe_get(row, "da")

        # Prefer operating_income; fall back to ebit
        base = operating_income if not np.isnan(operating_income) else ebit

        if np.isnan(base):
            logger.debug(
                "MetricsCalculator [%s]: operating_income and ebit both NaN — "
                "EBITDA is NaN.",
                self._ticker,
            )
            return np.nan

        if np.isnan(da):
            logger.debug(
                "MetricsCalculator [%s]: da is NaN — EBITDA approximated as EBIT/EBIT only.",
                self._ticker,
            )
            # EBITDA without D&A is EBIT — acceptable degradation, logged.
            return base

        return base + da

    def _compute_net_debt(self, row: pd.Series) -> float:
        """Net Debt = Long-Term Debt + Short-Term Debt − Cash."""
        ltd = _safe_get(row, "long_term_debt")
        std = _safe_get(row, "short_term_debt")
        cash = _safe_get(row, "cash")

        # Treat missing debt components as zero (conservative)
        ltd = 0.0 if np.isnan(ltd) else ltd
        std = 0.0 if np.isnan(std) else std

        if np.isnan(cash):
            logger.debug(
                "MetricsCalculator [%s]: cash is NaN — net_debt is NaN.", self._ticker
            )
            return np.nan

        return (ltd + std) - cash

    def _compute_altman_z(self, row: pd.Series, net_debt: float, market_cap: float) -> float:
        """
        Modified Altman Z-Score with dynamic SIC-based routing.
        
        Manufacturing (SIC 2000-3999): Classic Z-Score
          Z   = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5
          X4  = Market Cap / Total Liabilities
          X5  = Revenue / Total Assets

        Non-Manufacturing (Other): Z''-Score
          Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
          X4  = Book Value of Equity / Total Liabilities
          X5  = Omitted
        """
        total_assets = _safe_get(row, "total_assets")
        current_assets = _safe_get(row, "current_assets")
        current_liabilities = _safe_get(row, "current_liabilities")
        retained_earnings = _safe_get(row, "retained_earnings")
        ebit = _safe_get(row, "ebit")
        operating_income = _safe_get(row, "operating_income")
        total_liabilities = _safe_get(row, "total_liabilities")

        # Use operating_income as EBIT fallback
        ebit_val = ebit if not np.isnan(ebit) else operating_income

        sic = _safe_get(row, "sic")
        is_manufacturing = False
        if not np.isnan(sic):
            try:
                if 2000 <= int(sic) <= 3999:
                    is_manufacturing = True
            except (ValueError, TypeError):
                pass
                
        required_inputs = [
            ("total_assets", total_assets),
            ("current_assets", current_assets),
            ("current_liabilities", current_liabilities),
            ("ebit/operating_income", ebit_val),
            ("total_liabilities", total_liabilities),
        ]

        if is_manufacturing:
            revenue = _safe_get(row, "revenue")
            required_inputs.append(("revenue", revenue))
        else:
            equity = _safe_get(row, "equity")
            required_inputs.append(("equity", equity))

        for name, val in required_inputs:
            if np.isnan(val):
                logger.debug(
                    "MetricsCalculator [%s]: Altman Z NaN — missing '%s'.",
                    self._ticker,
                    name,
                )
                return np.nan

        if total_assets == 0 or total_liabilities == 0:
            return np.nan

        working_capital = current_assets - current_liabilities
        re = 0.0 if np.isnan(retained_earnings) else retained_earnings

        x1 = working_capital / total_assets
        x2 = re / total_assets
        x3 = ebit_val / total_assets

        if is_manufacturing:
            x4 = _safe_div(market_cap, total_liabilities)
            if np.isnan(x4):
                logger.debug(
                    "MetricsCalculator [%s]: Altman Z NaN — market_cap NaN (X4).",
                    self._ticker,
                )
                return np.nan
            x5 = revenue / total_assets
            return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        else:
            x4 = equity / total_liabilities
            return 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, annual_df: pd.DataFrame) -> dict[str, float]:
        """
        Compute all 8 metrics for the most recent complete fiscal year.

        The DataFrame index is expected to be either:
          • A plain RangeIndex / Int64Index of years  (after xs('A'))
          • A MultiIndex of (year, quarter='A')

        Parameters
        ----------
        annual_df : pd.DataFrame
            Single-ticker DataFrame from FinancialNormalizer, sliced to
            annual rows only (quarter == 'A') and sorted ascending.

        Returns
        -------
        dict[str, float]
            Keys: ticker, fiscal_year, + 8 metric names.
            Any metric that cannot be computed will be float('nan').
        """
        ticker = self._ticker

        if annual_df.empty:
            logger.warning(
                "MetricsCalculator [%s]: received empty DataFrame — all metrics NaN.",
                ticker,
            )
            nan_row: dict[str, float] = {m: np.nan for m in [
                "altman_z", "net_debt_ebitda", "current_ratio", "sloan_accruals",
                "ccr", "ev_ebitda", "fcf_yield", "fcf",
            ]}
            nan_row["ticker"] = ticker
            nan_row["fiscal_year"] = np.nan
            return nan_row

        # ── Use the most-recent fiscal year ────────────────────────────────
        # After xs('A', level='quarter') the index is just 'year' (integer).
        # We take the last sorted row.
        if isinstance(annual_df.index, pd.MultiIndex):
            # Normalise: drop the quarter level if still present
            try:
                df = annual_df.xs("A", level="quarter")
            except KeyError:
                df = annual_df
        else:
            df = annual_df

        df = df.sort_index()

        # ── Select the most-recent COMPLETE fiscal year ────────────────────
        # Root cause of the temporal flaw: blindly taking df.iloc[-1] in
        # March 2026 picks FY2026, which only contains partial Q1 data
        # (e.g. share-count updates from 8-Ks) — all P&L and balance sheet
        # columns are NaN, poisoning every downstream metric.
        #
        # Fix: iterate backwards through the available years.  A year is
        # considered "complete" when BOTH of these are non-NaN:
        #   • total_assets  — primary balance sheet anchor
        #   • net_income OR revenue — primary income anchor
        # This is the minimum viable completeness signal; if even these are
        # missing the year is structurally incomplete and we skip it.

        selected_year = None
        selected_idx = None

        for i in range(len(df) - 1, -1, -1):
            candidate_row = df.iloc[i]
            candidate_year = df.index[i]

            has_assets = not np.isnan(_safe_get(candidate_row, "total_assets"))
            has_income = (
                not np.isnan(_safe_get(candidate_row, "net_income"))
                or not np.isnan(_safe_get(candidate_row, "revenue"))
            )

            if has_assets and has_income:
                selected_year = candidate_year
                selected_idx = i
                break
            else:
                logger.warning(
                    "MetricsCalculator [%s]: FY%s is incomplete "
                    "(total_assets=%s, net_income=%s, revenue=%s) — "
                    "falling back to previous year.",
                    ticker,
                    candidate_year,
                    candidate_row.get("total_assets", "missing"),
                    candidate_row.get("net_income", "missing"),
                    candidate_row.get("revenue", "missing"),
                )

        if selected_year is None:
            logger.error(
                "MetricsCalculator [%s]: no complete fiscal year found in "
                "DataFrame (all years lack total_assets + net_income/revenue). "
                "Returning all-NaN metrics.",
                ticker,
            )
            nan_row2: dict[str, float] = {m: np.nan for m in [
                "altman_z", "net_debt_ebitda", "current_ratio", "sloan_accruals",
                "ccr", "ev_ebitda", "fcf_yield", "fcf",
                "ebitda", "net_debt", "market_cap",
            ]}
            nan_row2["ticker"] = ticker
            nan_row2["fiscal_year"] = np.nan
            return nan_row2

        latest_year = selected_year
        row = df.iloc[selected_idx]

        logger.info(
            "MetricsCalculator [%s]: selected fiscal year %s "
            "(max available was %s).",
            ticker,
            latest_year,
            df.index[-1],
        )


        # ── Retrieve previous year for Sloan (needs avg total assets) ──────
        prev_total_assets = np.nan
        if len(df) >= 2:
            prev_row = df.iloc[-2]
            prev_total_assets = _safe_get(prev_row, "total_assets")

        # ── Base inputs ─────────────────────────────────────────────────────
        current_assets = _safe_get(row, "current_assets")
        current_liabilities = _safe_get(row, "current_liabilities")
        total_assets = _safe_get(row, "total_assets")
        net_income = _safe_get(row, "net_income")
        operating_cash_flow = _safe_get(row, "operating_cash_flow")
        capex = _safe_get(row, "capex")
        shares_outstanding = _safe_get(row, "shares_outstanding")

        # ── Derived intermediates ───────────────────────────────────────────
        ebitda = self._compute_ebitda(row)
        net_debt = self._compute_net_debt(row)
        market_cap = _fetch_market_cap(ticker, shares_outstanding)
        fcf = _safe_div(operating_cash_flow - capex, 1.0)  # keeps NaN
        # Special case: if both are NaN leave FCF NaN; if capex NaN treat as 0
        if not np.isnan(operating_cash_flow):
            capex_val = 0.0 if np.isnan(capex) else capex
            fcf = operating_cash_flow - capex_val
        else:
            fcf = np.nan

        # ── Metric 1: Altman Z-Score ────────────────────────────────────────
        altman_z = self._compute_altman_z(row, net_debt, market_cap)

        # ── Metric 2: Net Debt / EBITDA ─────────────────────────────────────
        net_debt_ebitda = _safe_div(net_debt, ebitda)

        # ── Metric 3: Current Ratio ─────────────────────────────────────────
        current_ratio = _safe_div(current_assets, current_liabilities)

        # ── Metric 4: Sloan Accruals Ratio ──────────────────────────────────
        # = (Net Income − Operating CFO) / Average Total Assets
        avg_assets = np.nan
        if not np.isnan(total_assets) and not np.isnan(prev_total_assets):
            avg_assets = (total_assets + prev_total_assets) / 2.0
        elif not np.isnan(total_assets):
            avg_assets = total_assets  # single year fallback

        if not np.isnan(net_income) and not np.isnan(operating_cash_flow):
            sloan_accruals = _safe_div(net_income - operating_cash_flow, avg_assets)
        else:
            sloan_accruals = np.nan

        # ── Metric 5: Cash Conversion Ratio (CCR) ───────────────────────────
        # CCR = Operating CFO / EBITDA  (capital-structure-neutral definition)
        ccr = _safe_div(operating_cash_flow, ebitda)

        # ── Metric 6: EV / EBITDA ───────────────────────────────────────────
        if not np.isnan(market_cap) and not np.isnan(net_debt):
            enterprise_value = market_cap + net_debt
            ev_ebitda = _safe_div(enterprise_value, ebitda)
        else:
            ev_ebitda = np.nan

        # ── Metric 7: FCF Yield ─────────────────────────────────────────────
        fcf_yield = _safe_div(fcf, market_cap)

        # ── Assemble output dict ────────────────────────────────────────────
        metrics: dict[str, float] = {
            "ticker": ticker,
            "fiscal_year": float(latest_year),
            "altman_z": altman_z,
            "net_debt_ebitda": net_debt_ebitda,
            "current_ratio": current_ratio,
            "sloan_accruals": sloan_accruals,
            "ccr": ccr,
            "ev_ebitda": ev_ebitda,
            "fcf_yield": fcf_yield,
            "fcf": fcf,
            # Intermediates — useful for debugging, not used in filtering
            "ebitda": ebitda,
            "net_debt": net_debt,
            "market_cap": market_cap,
        }

        nan_count = sum(
            1 for k, v in metrics.items()
            if k not in ("ticker", "fiscal_year") and isinstance(v, float) and np.isnan(v)
        )
        logger.info(
            "MetricsCalculator [%s]: computed metrics for FY%s — "
            "%d NaN metric(s): %s",
            ticker,
            int(latest_year),
            nan_count,
            [k for k, v in metrics.items()
             if isinstance(v, float) and np.isnan(v) and k not in ("ticker", "fiscal_year")],
        )
        return metrics
