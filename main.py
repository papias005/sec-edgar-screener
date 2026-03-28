"""
main.py
-------
Primary entry point for the SEC EDGAR Quantitative Screener — Phase 2.

Pipeline flow
-------------
  EdgarClient
      ↓
  TickerMapper              (CIK + lazy SIC fetch)
      ↓
  DataExtractor             (raw XBRL JSON, disk-cached)
      ↓
  FinancialNormalizer       (XBRL → MultiIndex DataFrame)
      ↓
  MetricsCalculator         (8 composite metrics + live price via yfinance)
      ↓
  QuantitativeScreener      (Phase 1 hard filters → Phase 2 EV/EBITDA quintile)
      ↓
  Clean filtered DataFrame  (printed to stdout)

Usage
-----
    python main.py --tickers AAPL MSFT NVDA AMZN GOOGL

Environment variables
---------------------
    SEC_UA_NAME             Your name  (default: PlaceholderName)
    SEC_UA_EMAIL            Your email (default: placeholder@email.com)
    SEC_TICKER_CACHE        Disk cache path for tickers JSON
    SEC_RAW_CACHE_DIR       Directory for raw XBRL blobs
    SEC_INTER_REQUEST_DELAY Seconds between SEC API calls (default: 0.15)
    SEC_RATE_LIMIT_COOLDOWN Seconds to pause on 429/403  (default: 600)
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from edgar_client import EdgarClient
from ticker_mapper import TickerMapper
from data_extractor import DataExtractor
from financial_normalizer import FinancialNormalizer
from metrics_calculator import MetricsCalculator
from quantitative_screener import QuantitativeScreener

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    tickers: list[str],
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Run the complete Phase 2 pipeline over the provided ticker universe.

    Parameters
    ----------
    tickers : list[str]
        Stock tickers to process.
    force_refresh : bool
        If True, bypass disk caches and re-fetch all data from the SEC API.

    Returns
    -------
    pd.DataFrame
        Final filtered DataFrame (one row per surviving ticker, sorted by
        EV/EBITDA ascending).  Empty if no tickers survive the gauntlet.
    """
    log = logging.getLogger(__name__)
    log.info("=" * 70)
    log.info("  SEC EDGAR QUANTITATIVE SCREENER — PIPELINE START")
    log.info("=" * 70)
    log.info("Universe: %s (%d tickers)", ", ".join(tickers), len(tickers))

    # ── Shared infrastructure ─────────────────────────────────────────────
    client = EdgarClient()
    mapper = TickerMapper(client=client)
    extractor = DataExtractor(client=client, mapper=mapper)
    normalizer = FinancialNormalizer()

    metrics_list: list[dict] = []

    # ── Per-ticker processing ─────────────────────────────────────────────
    for ticker in tickers:
        log.info("── Processing %s ──────────────────────────────────────", ticker)

        # ── Step 1: Fetch SIC code (lazy from SEC submissions JSON) ───────
        sic: int | None = None
        try:
            sic = mapper.get_sic(ticker)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: could not fetch SIC: %s", ticker, exc)

        # ── Step 2: Download raw XBRL JSON ────────────────────────────────
        try:
            raw_json = extractor.fetch(ticker, force_refresh=force_refresh)
        except KeyError as exc:
            log.error("%s: ticker not found in SEC registry — %s", ticker, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            log.exception("%s: error fetching XBRL data — %s", ticker, exc)
            continue

        # ── Step 3: Normalise to DataFrame ────────────────────────────────
        df_full = normalizer.normalize(raw_json, ticker=ticker)
        if df_full.empty:
            log.warning("%s: normalised DataFrame is empty — skipping.", ticker)
            continue

        # ── Step 4: Slice to annual rows (quarter == 'A') ─────────────────
        try:
            if "quarter" in df_full.index.names:
                df_annual = df_full.xs("A", level="quarter")
            else:
                df_annual = df_full
        except KeyError:
            log.warning("%s: no annual (10-K) rows found — skipping.", ticker)
            continue

        if df_annual.empty:
            log.warning("%s: empty after annual slice — skipping.", ticker)
            continue

        log.info(
            "%s: annual data shape=%s, years=%s–%s",
            ticker,
            df_annual.shape,
            df_annual.index.min(),
            df_annual.index.max(),
        )

        # ── Step 5: Compute 8 composite metrics via MetricsCalculator ────
        calc = MetricsCalculator(ticker)
        metrics = calc.compute(df_annual)

        # Attach SIC code (needed by sector exclusion filter)
        metrics["sic"] = float(sic) if sic is not None else float("nan")

        metrics_list.append(metrics)
        log.info(
            "%s: metrics computed — altman_z=%.3f, cr=%.3f, ccr=%.3f, "
            "ev_ebitda=%.3f, fcf_yield=%.4f",
            ticker,
            metrics.get("altman_z", float("nan")),
            metrics.get("current_ratio", float("nan")),
            metrics.get("ccr", float("nan")),
            metrics.get("ev_ebitda", float("nan")),
            metrics.get("fcf_yield", float("nan")),
        )

    if not metrics_list:
        log.warning("Pipeline: no metrics computed — returning empty DataFrame.")
        return pd.DataFrame()

    # ── Step 6: Build cross-ticker universe DataFrame ─────────────────────
    universe_df = QuantitativeScreener.build_universe_df(metrics_list)
    log.info(
        "Pipeline: universe DataFrame built — %d tickers × %d columns.",
        len(universe_df),
        len(universe_df.columns),
    )

    # Export full unfiltered universe before any hard filters drop rows
    try:
        universe_df.to_csv("screener_full_audit.csv", index=True)
        log.info("Pipeline: Full unfiltered universe exported to screener_full_audit.csv")
    except Exception as exc:  # noqa: BLE001
        log.error("Pipeline: failed to export screener_full_audit.csv — %s", exc)

    # ── Step 7: Apply quantitative gauntlet ───────────────────────────────
    screener = QuantitativeScreener()
    result_df = screener.screen(universe_df)

    log.info("=" * 70)
    log.info(
        "  PIPELINE COMPLETE — %d/%d tickers survived the gauntlet.",
        len(result_df),
        len(tickers),
    )
    log.info("=" * 70)

    # Print audit log
    audit = screener.audit_log()
    if not audit.empty:
        log.info(
            "Audit log:\n%s",
            audit.to_string(index=False),
        )

    return result_df


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_results(result_df: pd.DataFrame) -> None:
    """Print the final screened universe to stdout in a readable table."""
    print()
    print("=" * 80)
    print("  QUANTITATIVE SCREENER — FINAL RESULTS")
    print("=" * 80)

    if result_df.empty:
        print("  No tickers survived all filters.")
        print("  Review the pipeline log for per-filter elimination counts.")
        return

    print(f"  {len(result_df)} ticker(s) passed the full quantitative gauntlet:\n")

    # Format floats for display
    display = result_df.copy()

    fmt_map: dict[str, str] = {
        "fiscal_year": "{:.0f}",
        "sic": "{:.0f}",
        "altman_z": "{:.3f}",
        "net_debt_ebitda": "{:.3f}",
        "current_ratio": "{:.3f}",
        "sloan_accruals": "{:.4f}",
        "ccr": "{:.4f}",
        "ev_ebitda": "{:.3f}",
        "fcf_yield": "{:.4f}",
        "fcf": "{:,.0f}",
        "ebitda": "{:,.0f}",
        "net_debt": "{:,.0f}",
        "market_cap": "{:,.0f}",
    }
    for col, fmt in fmt_map.items():
        if col in display.columns:
            display[col] = display[col].apply(
                lambda v, f=fmt: f.format(v) if not pd.isna(v) else "NaN"  # type: ignore[arg-type]
            )

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    pd.set_option("display.colheader_justify", "right")
    print(display.to_string())
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEC EDGAR Quantitative Screener — Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python main.py --tickers AAPL MSFT NVDA AMZN GOOGL META\n\n"
            "Set SEC_UA_NAME and SEC_UA_EMAIL environment variables to\n"
            "identify yourself to the SEC API (mandatory per ToS)."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        metavar="TICKER",
        help="One or more stock tickers to evaluate.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Bypass disk caches and re-fetch all data from the SEC API.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    result_df = run_pipeline(
        tickers=[t.upper() for t in args.tickers],
        force_refresh=args.force_refresh,
    )
    print_results(result_df)


if __name__ == "__main__":
    main()
