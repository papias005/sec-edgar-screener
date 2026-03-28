"""
edgar_client.py
---------------
Core authenticated, rate-limited HTTP client for the SEC EDGAR API.

Compliance notes:
  • Max 10 req/s mandated by SEC. We target ≤ 6.6 req/s (sleep 0.15 s between calls).
  • User-Agent header is mandatory; default Python UA is blocked.
  • HTTP 429 / 403 → exponential back-off, then a mandatory 600-second cooldown.
"""

import logging
import os
import time
from threading import Lock

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration (override via environment variables or pass a dict)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "user_agent_name": os.environ.get("SEC_UA_NAME", "PlaceholderName"),
    "user_agent_email": os.environ.get("SEC_UA_EMAIL", "placeholder@email.com"),
    # seconds between every request – keeps us at ~6.6 req/s
    "inter_request_delay": float(os.environ.get("SEC_INTER_REQUEST_DELAY", "0.15")),
    # seconds to cool-down after receiving a 429 / 403
    "rate_limit_cooldown": int(os.environ.get("SEC_RATE_LIMIT_COOLDOWN", "600")),
    # exponential back-off base (seconds) before the big cooldown
    "backoff_base": float(os.environ.get("SEC_BACKOFF_BASE", "2.0")),
    "max_retries": int(os.environ.get("SEC_MAX_RETRIES", "5")),
    "request_timeout": int(os.environ.get("SEC_REQUEST_TIMEOUT", "30")),
}


class RateLimiter:
    """Thread-safe token-bucket style rate limiter via a simple sleep gate."""

    def __init__(self, min_interval: float):
        """
        Parameters
        ----------
        min_interval : float
            Minimum seconds to wait between consecutive requests.
        """
        self._min_interval = min_interval
        self._last_call_time: float = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                logger.debug("RateLimiter: sleeping %.3f s", sleep_for)
                time.sleep(sleep_for)
            self._last_call_time = time.monotonic()


class EdgarClient:
    """
    Authenticated, rate-limited HTTP client for SEC EDGAR endpoints.

    Usage
    -----
    >>> client = EdgarClient()
    >>> data = client.get_json("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json")
    """

    def __init__(self, config: dict | None = None):
        self._cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._rate_limiter = RateLimiter(self._cfg["inter_request_delay"])
        self._session = self._build_session()
        logger.info(
            "EdgarClient initialised – User-Agent: '%s <%s>'",
            self._cfg["user_agent_name"],
            self._cfg["user_agent_email"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    f"{self._cfg['user_agent_name']} "
                    f"{self._cfg['user_agent_email']}"
                ),
                "Accept-Encoding": "gzip, deflate",
                # NOTE: Do NOT set 'Host' at the session level.
                # requests sets it automatically per-request based on the URL,
                # so hardcoding it here would break calls to www.sec.gov.
            }
        )
        # Low-level TCP retry for transient connectivity failures (not 4xx/5xx)
        retry_policy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_policy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _handle_rate_limit(self, attempt: int) -> None:
        """
        Exponential back-off for 429/403 responses.
        On the final attempt threshold, enforces the full 600-second cooldown.
        """
        if attempt >= self._cfg["max_retries"] - 1:
            cooldown = self._cfg["rate_limit_cooldown"]
            logger.warning(
                "Rate-limit cooldown triggered. Pausing for %d seconds (%.1f min).",
                cooldown,
                cooldown / 60,
            )
            time.sleep(cooldown)
        else:
            backoff = self._cfg["backoff_base"] ** attempt
            logger.warning(
                "HTTP 429/403 received (attempt %d). Back-off %.1f s.", attempt, backoff
            )
            time.sleep(backoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Perform a GET request with mandatory rate limiting and fault tolerance.

        Retries on HTTP 429 and 403 with exponential back-off, culminating in
        a mandatory 600-second cooldown before raising an exception.

        Parameters
        ----------
        url : str
            The full URL to fetch.
        **kwargs
            Passed through to requests.Session.get().

        Returns
        -------
        requests.Response

        Raises
        ------
        requests.HTTPError
            If all retry attempts are exhausted.
        """
        kwargs.setdefault("timeout", self._cfg["request_timeout"])

        for attempt in range(self._cfg["max_retries"]):
            self._rate_limiter.wait()
            logger.debug("GET %s (attempt %d)", url, attempt + 1)

            try:
                response = self._session.get(url, **kwargs)
            except requests.RequestException as exc:
                logger.error("Network error on GET %s: %s", url, exc)
                if attempt < self._cfg["max_retries"] - 1:
                    time.sleep(self._cfg["backoff_base"] ** attempt)
                    continue
                raise

            if response.status_code in (429, 403):
                logger.warning("HTTP %d from %s", response.status_code, url)
                self._handle_rate_limit(attempt)
                continue

            if response.status_code == 200:
                logger.info("GET %s → 200 OK", url)
                return response

            # Non-retriable HTTP error
            logger.error("HTTP %d from %s", response.status_code, url)
            response.raise_for_status()

        raise requests.HTTPError(
            f"All {self._cfg['max_retries']} attempts exhausted for {url}"
        )

    def get_json(self, url: str, **kwargs) -> dict:
        """Convenience wrapper that parses the JSON response body."""
        response = self.get(url, **kwargs)
        return response.json()
