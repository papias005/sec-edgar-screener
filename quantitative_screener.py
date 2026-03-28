"""
quantitative_screener.py
------------------------
Cross-ticker quantitative screening engine.

Architecture
------------
Unlike PipelineScreener (which operates on a single ticker's time-series rows),
QuantitativeScreener operates on the **cross-sectional universe** — one row
per ticker, each row containing that ticker's computed 8 metrics.

The screening is a two-phase gauntlet:

  Phase 1 — Per-row hard exclusions (applied sequentially):
    All 8 filters from screener_filters.ALL_HARD_FILTERS are applied one at
    a time. Any ticker failing a filter is dropped immediately with a log entry
    explaining which filter eliminated it. NaN in any critical metric triggers
    automatic exclusion.

  Phase 2 — Cross-sectional EV/EBITDA quintile filter:
    Among the survivors of Phase 1, only tickers whose EV/EBITDA falls in the
    *bottom quintile* (cheapest 20% by valuation) are retained.
    This is the only relative filter — it requires the full surviving universe
    to be present before it can be applied.

Input
-----
A pandas.DataFrame with:
  • Index: ticker symbols (str)
  • Columns: all 8 metrics + 'sic', 'fiscal_year', 'ebitda', 'net_debt',
    'market_cap' (as produced by collecting MetricsCalculator.compute()
    outputs and calling QuantitativeScreener.build_universe_df()).

Output
------
A clean, filtered pandas.DataFrame containing only the tickers that survived
both phases, sorted ascending by EV/EBITDA (best value first).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from screener_filters import ALL_HARD_FILTERS

logger = logging.getLogger(__name__)

# Bottom quintile = lowest 20% of EV/EBITDA values (cheapest valuation)
EV_EBITDA_QUINTILE_CUTOFF: float = 0.20


# Display columns in the final output (in this order)
_OUTPUT_COLUMNS: list[str] = [
    "fiscal_year",
    "sic",
    "altman_z",
    "net_debt_ebitda",
    "current_ratio",
    "sloan_accruals",
    "ccr",
    "ev_ebitda",
    "fcf_yield",
    "fcf",
    "ebitda",
    "net_debt",
    "market_cap",
]


class QuantitativeScreener:
    """
    Cross-ticker quantitative screening engine.

    Parameters
    ----------
    filters : list[tuple[str, callable]] | None
        Ordered list of (name, filter_fn) tuples.  Each filter_fn accepts a
        pd.Series (one row of the universe DataFrame) and returns bool.
        Defaults to ALL_HARD_FILTERS from screener_filters.
    ev_ebitda_quintile : float
        Fraction defining the 'cheap' valuation cutoff.
        0.20 = bottom 20% (default).  Must be in (0, 1].

    Examples
    --------
    >>> screener = QuantitativeScreener()
    >>> result_df = screener.screen(universe_df)
    """

    def __init__(
        self,
        filters: list[tuple[str, callable]] | None = None,
        ev_ebitda_quintile: float = EV_EBITDA_QUINTILE_CUTOFF,
    ):
        self._filters = filters if filters is not None else list(ALL_HARD_FILTERS)
        self._quintile = ev_ebitda_quintile
        self._audit_log: list[dict] = []  # populated during screen()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_universe_df(metrics_list: list[dict]) -> pd.DataFrame:
        """
        Convert a list of per-ticker metric dicts (from MetricsCalculator)
        into a cross-sectional DataFrame indexed by ticker.

        Parameters
        ----------
        metrics_list : list[dict]
            Each dict must contain at least a 'ticker' key.

        Returns
        -------
        pd.DataFrame
            One row per ticker. Missing columns filled with NaN.
        """
        if not metrics_list:
            logger.warning(
                "QuantitativeScreener.build_universe_df: empty metrics_list."
            )
            return pd.DataFrame()

        df = pd.DataFrame(metrics_list)
        if "ticker" in df.columns:
            df = df.set_index("ticker")
        df.index.name = "ticker"
        logger.info(
            "QuantitativeScreener.build_universe_df: built universe with %d tickers, "
            "columns: %s",
            len(df),
            list(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def screen(self, universe_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full quantitative gauntlet to the universe DataFrame.

        Phase 1: sequential per-row hard filters (any fail → drop).
        Phase 2: EV/EBITDA bottom-quintile filter on survivors.

        Parameters
        ----------
        universe_df : pd.DataFrame
            Cross-sectional universe.  Index = tickers, columns = metrics.

        Returns
        -------
        pd.DataFrame
            Tickers that passed all filters, sorted ascending by EV/EBITDA.
            Empty DataFrame if no tickers survive.
        """
        if universe_df.empty:
            logger.warning("QuantitativeScreener.screen: universe is empty.")
            return pd.DataFrame()

        self._audit_log = []
        total_in = len(universe_df)
        logger.info(
            "QuantitativeScreener: BEGIN screening — %d tickers in universe.", total_in
        )

        surviving = universe_df.copy()

        # ── Phase 1: Vectorized DataFrame processing ─────────────────────────
        for filter_name, filter_fn in self._filters:
            before = len(surviving)

            try:
                pass_mask = filter_fn(surviving)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "QuantitativeScreener: filter '%s' raised exception: %s "
                    "— treating all as EXCLUDED.",
                    filter_name,
                    exc,
                )
                pass_mask = pd.Series(False, index=surviving.index)

            pass_mask = pass_mask.fillna(False).astype(bool)

            # Log exclusions
            excluded = surviving[~pass_mask]
            for ticker_idx in excluded.index:
                logger.debug("%s [%s]: EXCLUDED.", filter_name, ticker_idx)

            surviving = surviving.loc[pass_mask]
            after = len(surviving)
            eliminated = before - after

            self._audit_log.append({
                "filter": filter_name,
                "before": before,
                "after": after,
                "eliminated": eliminated,
            })
            logger.info(
                "QuantitativeScreener | %-22s | %3d → %3d (%d eliminated)",
                filter_name,
                before,
                after,
                eliminated,
            )

            if surviving.empty:
                logger.warning(
                    "QuantitativeScreener: all tickers eliminated after filter '%s'.",
                    filter_name,
                )
                return pd.DataFrame()

        # ── Phase 2: EV/EBITDA quintile filter ──────────────────────────────
        before_quintile = len(surviving)

        if "ev_ebitda" not in surviving.columns:
            logger.warning(
                "QuantitativeScreener: 'ev_ebitda' column missing — "
                "skipping quintile filter."
            )
        else:
            ev_series = surviving["ev_ebitda"].dropna()

            if ev_series.empty:
                logger.warning(
                    "QuantitativeScreener: all EV/EBITDA values are NaN after Phase 1 "
                    "— quintile filter cannot be applied, returning Phase 1 survivors."
                )
            else:
                cutoff_value = ev_series.quantile(self._quintile)
                logger.info(
                    "QuantitativeScreener: EV/EBITDA bottom-%.0f%% cutoff = %.4f",
                    self._quintile * 100,
                    cutoff_value,
                )

                # Keep only tickers with EV/EBITDA ≤ cutoff (bottom quintile = cheapest)
                quintile_mask = surviving["ev_ebitda"].fillna(np.inf) <= cutoff_value
                surviving = surviving.loc[quintile_mask]

        after_quintile = len(surviving)
        self._audit_log.append({
            "filter": f"ev_ebitda_quintile (bottom {self._quintile*100:.0f}%)",
            "before": before_quintile,
            "after": after_quintile,
            "eliminated": before_quintile - after_quintile,
        })
        logger.info(
            "QuantitativeScreener | %-22s | %3d → %3d (%d eliminated)",
            f"ev_ebitda_quintile_{int(self._quintile*100)}pct",
            before_quintile,
            after_quintile,
            before_quintile - after_quintile,
        )

        # ── Final sort: best value (lowest EV/EBITDA) first ─────────────────
        if "ev_ebitda" in surviving.columns and not surviving.empty:
            surviving = surviving.sort_values("ev_ebitda", ascending=True)

        # ── Log final result ─────────────────────────────────────────────────
        logger.info(
            "QuantitativeScreener: COMPLETE — %d/%d tickers passed all filters.",
            len(surviving),
            total_in,
        )
        self._log_audit_summary()

        # ── Return clean output columns (subset, ordered) ────────────────────
        out_cols = [c for c in _OUTPUT_COLUMNS if c in surviving.columns]
        return surviving[out_cols] if out_cols else surviving

    def audit_log(self) -> pd.DataFrame:
        """
        Return a DataFrame showing how many tickers were eliminated at each
        filter stage.  Populated after calling screen().
        """
        if not self._audit_log:
            return pd.DataFrame(columns=["filter", "before", "after", "eliminated"])
        return pd.DataFrame(self._audit_log)

    def _log_audit_summary(self) -> None:
        """Emit a formatted audit table to the logger at INFO level."""
        if not self._audit_log:
            return
        width = 68
        logger.info("=" * width)
        logger.info("  QUANTITATIVE SCREENER — AUDIT SUMMARY")
        logger.info("=" * width)
        logger.info("  %-30s %8s %8s %10s", "Filter", "Before", "After", "Dropped")
        logger.info("  " + "-" * (width - 2))
        for entry in self._audit_log:
            logger.info(
                "  %-30s %8d %8d %10d",
                entry["filter"][:30],
                entry["before"],
                entry["after"],
                entry["eliminated"],
            )
        logger.info("=" * width)
