# SEC EDGAR Quantitative Screener (US-GAAP)

A production-grade algorithmic screening pipeline built in Python to extract, normalize, and evaluate financial data directly from the SEC EDGAR REST API. The system identifies undervalued, fundamentally resilient small-cap equities by parsing raw XBRL corporate filings and merging them with real-time market data.

## System Architecture

1. **Data Extraction (`edgar_client.py`, `data_extractor.py`)**
   - Bypasses commercial data aggregators by fetching raw `CompanyFacts` JSON payloads directly from the SEC.
   - Enforces strict ToS compliance: `0.15s` rate-limiting, exponential backoff for HTTP 429/403, and mandatory User-Agent header spoofing.
   - Implements asynchronous local disk caching to prevent `MemoryError` and minimize network latency across large index scans.

2. **XBRL Parsing & Normalization (`financial_normalizer.py`)**
   - Maps disjointed US-GAAP XBRL tags into a standardized, multi-index Pandas DataFrame (Year, Quarter).
   - Implements duration-based deduplication and fallback arrays to resolve reporting inconsistencies across 10-K and 10-Q filings.

3. **Quantitative Engine (`metrics_calculator.py`)**
   - Computes 8 absolute financial metrics (Net Debt/EBITDA, FCF Yield, Sloan Accruals, Cash Conversion Ratio, etc.).
   - Integrates `yfinance` to inject real-time asset pricing, calculating live Enterprise Value (EV) and Market Capitalization.

4. **Dynamic Sector Routing (`screener_filters.py`)**
   - **SIC-Based Bifurcation:** Dynamically routes bankruptcy calculations based on the Standard Industrial Classification (SIC) code.
   - Applies the **Classic Altman Z-Score** (> 2.99) for manufacturing/heavy industry.
   - Applies the **Altman Z''-Score** (> 2.60) for asset-light, retail, and service sectors, explicitly removing the Asset Turnover penalty to prevent systemic Type I errors.

## Tech Stack
`Python 3.10+` | `Pandas` | `NumPy` | `Requests` | `yfinance`

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
2. Environment Variables
You must declare your identity to the SEC EDGAR system to avoid permanent IP bans.

PowerShell
$env:SEC_UA_NAME="YourName"
$env:SEC_UA_EMAIL="your.email@domain.com"
3. Execution
Run the pipeline against specific tickers or an entire index. The system will output a tabular summary and export a comprehensive screener_full_audit.csv file.

Bash
python main.py --tickers AAPL MSFT NVDA AMZN --log-level INFO
