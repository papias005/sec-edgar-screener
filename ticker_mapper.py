"""
ticker_mapper.py
----------------
Maps standard stock tickers (e.g. 'AAPL') to their SEC 10-digit CIK numbers
and Standard Industrial Classification (SIC) codes.

Phase 2 change: now fetches company_tickers_exchange.json instead of
company_tickers.json — the exchange variant includes SIC codes, which are
required for sector-exclusion filtering.

SEC endpoint:
  https://www.sec.gov/files/company_tickers_exchange.json

The rate limiter in EdgarClient applies to this fetch.
"""

import json
import logging
import os
from pathlib import Path

from edgar_client import EdgarClient

logger = logging.getLogger(__name__)

# Use the exchange variant — it carries SIC codes.
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

DEFAULT_CACHE_PATH = Path(
    os.environ.get("SEC_TICKER_CACHE", "cache/company_tickers_exchange.json")
)


class TickerMapper:
    """
    Fetches and caches the SEC company_tickers_exchange.json for fast CIK
    and SIC code lookups.

    The exchange JSON format is:
      {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data":   [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]
      }

    Note: SIC codes are NOT present in company_tickers_exchange.json.
    We retrieve SIC codes lazily per-company from the SEC EDGAR company
    search endpoint:
      https://data.sec.gov/submissions/CIK{10_digit_cik}.json

    SIC codes are cached in-memory once fetched.

    Parameters
    ----------
    client : EdgarClient
        A pre-configured, rate-limited EdgarClient instance.
    cache_path : Path | None
        Optional path to the disk-cached tickers JSON.

    Examples
    --------
    >>> mapper = TickerMapper(client)
    >>> mapper.get_cik("AAPL")
    '0000320193'
    >>> mapper.get_sic("AAPL")
    3674
    """

    _SUBMISSIONS_BASE = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(
        self,
        client: EdgarClient,
        cache_path: Path | None = DEFAULT_CACHE_PATH,
    ):
        self._client = client
        self._cache_path = cache_path

        # { "AAPL": "0000320193" }
        self._ticker_to_cik: dict[str, str] = {}
        # In-memory SIC cache — populated lazily per ticker
        # { "AAPL": 3674 }
        self._ticker_to_sic: dict[str, int | None] = {}

        self._load()

    # ------------------------------------------------------------------
    # Private helpers — ticker/CIK loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Populate the in-memory map from disk cache or the SEC API."""
        if self._cache_path and self._cache_path.exists():
            self._load_from_disk()
        else:
            self._fetch_from_api()

    def _load_from_disk(self) -> None:
        logger.info("TickerMapper: loading from disk cache %s", self._cache_path)
        with open(self._cache_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._parse_raw(raw)

    def _fetch_from_api(self) -> None:
        logger.info("TickerMapper: fetching %s", SEC_TICKERS_URL)
        raw = self._client.get_json(SEC_TICKERS_URL)
        self._parse_raw(raw)
        if self._cache_path:
            self._save_to_disk(raw)

    def _parse_raw(self, raw: dict) -> None:
        """
        Parse the company_tickers_exchange.json payload.

        Expected format (list-of-rows with a 'fields' header):
          {
            "fields": ["cik", "name", "ticker", "exchange"],
            "data":   [[320193, "Apple Inc.", "AAPL", "Nasdaq"], ...]
          }

        Older / alternative format (dict-of-dicts, same as company_tickers.json):
          { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ... }

        Both are handled gracefully.
        """
        # Format A: list-based with a 'fields' header (exchange variant)
        if "fields" in raw and "data" in raw:
            fields = raw["fields"]
            try:
                cik_idx = fields.index("cik")
                ticker_idx = fields.index("ticker")
            except ValueError:
                logger.error(
                    "TickerMapper: unexpected 'fields' schema in tickers JSON: %s",
                    fields,
                )
                return
            for row in raw["data"]:
                try:
                    ticker = str(row[ticker_idx]).upper()
                    cik = str(row[cik_idx]).zfill(10)
                    if ticker and cik:
                        self._ticker_to_cik[ticker] = cik
                except (IndexError, TypeError):
                    continue

        # Format B: dict-of-dicts (company_tickers.json style fallback)
        elif isinstance(raw, dict):
            for entry in raw.values():
                if not isinstance(entry, dict):
                    continue
                ticker = str(entry.get("ticker", "")).upper()
                cik_raw = entry.get("cik_str", entry.get("cik", ""))
                if ticker and cik_raw:
                    self._ticker_to_cik[ticker] = str(cik_raw).zfill(10)

        logger.info(
            "TickerMapper: loaded %d ticker→CIK mappings.", len(self._ticker_to_cik)
        )

    def _save_to_disk(self, raw: dict) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w", encoding="utf-8") as fh:
            json.dump(raw, fh)
        logger.info("TickerMapper: cache written to %s", self._cache_path)

    # ------------------------------------------------------------------
    # Private helpers — SIC code fetching (lazy, per-ticker)
    # ------------------------------------------------------------------

    def _fetch_sic(self, cik: str) -> int | None:
        """
        Fetch the SIC code for a company from the SEC submissions endpoint.

        Returns the SIC code as an integer, or None if unavailable.
        The result is cached in self._ticker_to_sic.
        """
        url = self._SUBMISSIONS_BASE.format(cik=cik)
        logger.debug("TickerMapper: fetching SIC for CIK %s from %s", cik, url)
        try:
            data = self._client.get_json(url)
            sic = data.get("sic")
            return int(sic) if sic is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TickerMapper: could not fetch SIC for CIK %s: %s", cik, exc
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cik(self, ticker: str) -> str:
        """
        Return the zero-padded 10-digit CIK for a given ticker symbol.

        Parameters
        ----------
        ticker : str
            Standard stock ticker, e.g. 'AAPL', 'MSFT'.

        Returns
        -------
        str
            Zero-padded CIK, e.g. '0000320193'.

        Raises
        ------
        KeyError
            If the ticker is not found in the SEC registry.
        """
        key = ticker.upper()
        if key not in self._ticker_to_cik:
            raise KeyError(
                f"Ticker '{ticker}' not found in SEC company_tickers_exchange.json. "
                "Check spelling or try the full legal company name search."
            )
        cik = self._ticker_to_cik[key]
        logger.debug("TickerMapper: %s → CIK %s", ticker, cik)
        return cik

    def get_sic(self, ticker: str) -> int | None:
        """
        Return the Standard Industrial Classification (SIC) code for a ticker.

        The SIC code is fetched lazily from the SEC submissions JSON on first
        request and then cached in memory for subsequent calls.

        Parameters
        ----------
        ticker : str
            Standard stock ticker symbol.

        Returns
        -------
        int | None
            The integer SIC code (e.g. 3674 for semiconductors), or None if
            it cannot be determined.
        """
        key = ticker.upper()
        if key in self._ticker_to_sic:
            return self._ticker_to_sic[key]

        try:
            cik = self.get_cik(key)
        except KeyError:
            self._ticker_to_sic[key] = None
            return None

        sic = self._fetch_sic(cik)
        self._ticker_to_sic[key] = sic
        logger.info("TickerMapper: %s SIC = %s", ticker, sic)
        return sic

    def refresh(self) -> None:
        """Force a fresh fetch from the SEC API, bypassing the cache."""
        logger.info("TickerMapper: refreshing from API.")
        self._fetch_from_api()

    @property
    def all_tickers(self) -> list[str]:
        """Return a sorted list of all known ticker symbols."""
        return sorted(self._ticker_to_cik.keys())
