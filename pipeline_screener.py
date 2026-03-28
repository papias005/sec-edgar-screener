"""
pipeline_screener.py
--------------------
Base interface and concrete implementation for the PipelineScreener.

Design philosophy
-----------------
The screener is intentionally decoupled from the financial evaluation logic.
Concrete filter functions are injected at runtime – they are never hard-coded
here. This allows industry-specific or strategy-specific filter sets to be
added in a future phase without touching the core pipeline.

Filter contract
---------------
A filter is any callable with the signature:

    def my_filter(df: pd.DataFrame) -> pd.Series[bool]:
        \"\"\"Return a boolean mask aligned to df.index.\"\"\"
        ...

The mask marks rows (periods) that PASS the filter as True.

Usage
-----
>>> screener = PipelineScreener(df, ticker="AAPL")
>>> screener.add_filter(my_filter_a)
>>> screener.add_filter(my_filter_b)
>>> passing_df = screener.apply_filters()
"""

from __future__ import annotations

import abc
import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Type alias for a filter callable
FilterFn = Callable[[pd.DataFrame], "pd.Series[bool]"]


# ---------------------------------------------------------------------------
# Abstract base – defines the public interface
# ---------------------------------------------------------------------------


class BaseScreener(abc.ABC):
    """Abstract base class that every screener implementation must satisfy."""

    @abc.abstractmethod
    def add_filter(self, filter_fn: FilterFn, *, name: str | None = None) -> None:
        """Register a callable filter to be applied later."""

    @abc.abstractmethod
    def apply_filters(
        self,
        filter_list: list[FilterFn] | None = None,
    ) -> pd.DataFrame:
        """
        Apply all registered (and optionally additionally supplied) filters.

        Parameters
        ----------
        filter_list : list[FilterFn] | None
            Optional extra filters applied on top of those already registered
            via add_filter().  Useful for one-shot screening without mutating
            the screener's state.

        Returns
        -------
        pd.DataFrame
            The subset of the input DataFrame whose rows pass **all** filters.
        """

    @abc.abstractmethod
    def reset_filters(self) -> None:
        """Clear all registered filters."""

    @abc.abstractmethod
    def summary(self) -> dict:
        """Return a summary of the current screener state."""


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


class PipelineScreener(BaseScreener):
    """
    Production-grade screener that wraps a normalised financial DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The normalised DataFrame produced by FinancialNormalizer.normalize().
        Expected MultiIndex: (year, quarter).
    ticker : str
        Ticker label for logging; does not affect logic.
    """

    def __init__(self, data: pd.DataFrame, ticker: str = ""):
        self._ticker = ticker
        self._data: pd.DataFrame = data.copy()
        self._filters: list[tuple[str, FilterFn]] = []

        logger.info(
            "PipelineScreener [%s]: initialised with DataFrame shape %s",
            self._ticker or "?",
            self._data.shape,
        )

    # ------------------------------------------------------------------
    # BaseScreener implementation
    # ------------------------------------------------------------------

    def add_filter(self, filter_fn: FilterFn, *, name: str | None = None) -> None:
        """
        Register a filter callable.

        Parameters
        ----------
        filter_fn : FilterFn
            A callable ``(df: DataFrame) -> Series[bool]`` that returns a
            boolean mask aligned to df.index.
        name : str | None
            Human-readable name for logging. Defaults to the function's
            ``__name__`` attribute.
        """
        label = name or getattr(filter_fn, "__name__", repr(filter_fn))
        self._filters.append((label, filter_fn))
        logger.debug(
            "PipelineScreener [%s]: added filter '%s' (total: %d)",
            self._ticker,
            label,
            len(self._filters),
        )

    def apply_filters(
        self,
        filter_list: list[FilterFn] | None = None,
    ) -> pd.DataFrame:
        """
        Apply all registered + any ad-hoc filters and return passing rows.

        Each filter receives the **full** normalised DataFrame and must return
        a boolean Series with the **same index**.  Rows must pass *all*
        filters to appear in the result (logical AND across filters).

        Filters that raise exceptions are logged and treated as "pass-all"
        (to avoid silently dropping data due to a buggy filter).

        Parameters
        ----------
        filter_list : list[FilterFn] | None
            Additional one-shot filters on top of registered ones.

        Returns
        -------
        pd.DataFrame
            Filtered subset of the original DataFrame.
        """
        all_filters: list[tuple[str, FilterFn]] = list(self._filters)
        if filter_list:
            for fn in filter_list:
                label = getattr(fn, "__name__", repr(fn))
                all_filters.append((label, fn))

        if not all_filters:
            logger.warning(
                "PipelineScreener [%s]: no filters registered – returning full DataFrame.",
                self._ticker,
            )
            return self._data.copy()

        # Start with a mask that passes everything
        combined_mask = pd.Series(True, index=self._data.index)

        for label, filter_fn in all_filters:
            logger.debug(
                "PipelineScreener [%s]: applying filter '%s'", self._ticker, label
            )
            try:
                mask = filter_fn(self._data)
                if not isinstance(mask, pd.Series):
                    raise TypeError(
                        f"Filter '{label}' returned {type(mask).__name__}, expected pd.Series."
                    )
                if mask.index.to_list() != combined_mask.index.to_list():
                    # Re-align if index is a subset or has different ordering
                    mask = mask.reindex(combined_mask.index, fill_value=True)

                before = combined_mask.sum()
                combined_mask = combined_mask & mask.astype(bool)
                after = combined_mask.sum()
                logger.info(
                    "PipelineScreener [%s]: filter '%s' → %d→%d rows pass.",
                    self._ticker,
                    label,
                    before,
                    after,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "PipelineScreener [%s]: filter '%s' raised %s – treating as pass-all.",
                    self._ticker,
                    label,
                    exc,
                )

        result = self._data.loc[combined_mask]
        logger.info(
            "PipelineScreener [%s]: %d/%d rows passed all filters.",
            self._ticker,
            len(result),
            len(self._data),
        )
        return result

    def reset_filters(self) -> None:
        """Remove all registered filters."""
        self._filters.clear()
        logger.info("PipelineScreener [%s]: all filters cleared.", self._ticker)

    def summary(self) -> dict:
        """Return a dict summarising the screener's current configuration."""
        return {
            "ticker": self._ticker,
            "data_shape": self._data.shape,
            "columns": list(self._data.columns),
            "index_levels": list(self._data.index.names),
            "registered_filters": [name for name, _ in self._filters],
            "year_range": (
                (
                    self._data.index.get_level_values("year").min(),
                    self._data.index.get_level_values("year").max(),
                )
                if not self._data.empty
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Read-only access to the underlying normalised DataFrame."""
        return self._data.copy()

    def annual_data(self) -> pd.DataFrame:
        """Return only rows corresponding to annual (10-K) filings."""
        if self._data.empty:
            return self._data
        return self._data.xs("A", level="quarter")

    def quarterly_data(self) -> pd.DataFrame:
        """Return only rows corresponding to quarterly (10-Q) filings."""
        if self._data.empty:
            return self._data
        mask = self._data.index.get_level_values("quarter") != "A"
        return self._data.loc[mask]
