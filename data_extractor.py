"""
data_extractor.py
-----------------
Downloads the full XBRL / Company Facts JSON for a given company from the
SEC EDGAR API.

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{10_digit_cik}.json

The response is a large JSON blob (can be >10 MB for large filers) that
contains **all** US-GAAP and DEI facts filed since the company's first
10-K/10-Q on EDGAR.

This module keeps the raw JSON intact — parsing / normalisation lives in
FinancialNormalizer.
"""

import logging
from pathlib import Path
import json
import os

from edgar_client import EdgarClient
from ticker_mapper import TickerMapper

logger = logging.getLogger(__name__)

COMPANY_FACTS_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
DEFAULT_RAW_CACHE_DIR = Path(os.environ.get("SEC_RAW_CACHE_DIR", "cache/raw_facts"))


class DataExtractor:
    """
    Fetches the raw Company Facts JSON for a list of tickers / CIKs.

    Parameters
    ----------
    client : EdgarClient
        Authenticated, rate-limited HTTP client.
    mapper : TickerMapper
        Ticker-to-CIK resolver.
    cache_dir : Path | None
        Directory where raw JSON blobs are cached.  If None, caching is
        disabled and every call hits the SEC API.

    Examples
    --------
    >>> extractor = DataExtractor(client, mapper)
    >>> raw = extractor.fetch("AAPL")   # dict
    """

    def __init__(
        self,
        client: EdgarClient,
        mapper: TickerMapper,
        cache_dir: Path | None = DEFAULT_RAW_CACHE_DIR,
    ):
        self._client = client
        self._mapper = mapper
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, cik: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"CIK{cik}.json"

    def _load_from_cache(self, cik: str) -> dict | None:
        path = self._cache_path(cik)
        if path and path.exists():
            logger.info("DataExtractor: cache hit for CIK %s (%s)", cik, path)
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def _save_to_cache(self, cik: str, data: dict) -> None:
        path = self._cache_path(cik)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            logger.info("DataExtractor: cached raw facts for CIK %s → %s", cik, path)

    def _fetch_raw(self, cik: str) -> dict:
        url = f"{COMPANY_FACTS_BASE_URL}/CIK{cik}.json"
        logger.info("DataExtractor: downloading company facts for CIK %s", cik)
        return self._client.get_json(url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, *, force_refresh: bool = False) -> dict:
        """
        Return the raw Company Facts JSON for *ticker*.

        Parameters
        ----------
        ticker : str
            Standard stock ticker symbol.
        force_refresh : bool
            If True, bypass disk cache and re-fetch from the SEC API.

        Returns
        -------
        dict
            The full parsed XBRL Company Facts JSON.

        Raises
        ------
        KeyError
            If *ticker* is not found in the SEC registry.
        requests.HTTPError
            On unrecoverable HTTP errors from the SEC API.
        """
        cik = self._mapper.get_cik(ticker)
        logger.debug("DataExtractor.fetch: %s → CIK %s", ticker, cik)

        if not force_refresh:
            cached = self._load_from_cache(cik)
            if cached is not None:
                return cached

        raw = self._fetch_raw(cik)
        self._save_to_cache(cik, raw)
        return raw

    def fetch_batch(
        self,
        tickers: list[str],
        *,
        force_refresh: bool = False,
    ) -> dict[str, dict]:
        """
        Fetch Company Facts for multiple tickers sequentially.

        Returns
        -------
        dict[str, dict]
            Mapping of ticker → raw Company Facts JSON.
            Tickers that fail (e.g. not found, HTTP error) are logged and
            excluded from the result rather than crashing the entire batch.
        """
        results: dict[str, dict] = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch(ticker, force_refresh=force_refresh)
            except KeyError as exc:
                logger.warning("DataExtractor.fetch_batch: skipping %s – %s", ticker, exc)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "DataExtractor.fetch_batch: error fetching %s – %s", ticker, exc
                )
        logger.info(
            "DataExtractor.fetch_batch: completed %d/%d tickers.",
            len(results),
            len(tickers),
        )
        return results
