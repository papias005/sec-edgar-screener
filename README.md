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


Ποσοτικός Αναλυτής SEC EDGAR (US-GAAP)
Ένα αλγοριθμικό σύστημα παραγωγικής βαθμίδας (production-grade) σε Python για την απευθείας άντληση, κανονικοποίηση και ποσοτική αξιολόγηση χρηματοοικονομικών δεδομένων από το REST API της αμερικανικής Επιτροπής Κεφαλαιαγοράς (SEC EDGAR). Το σύστημα εντοπίζει υποτιμημένες, θεμελιωδώς ανθεκτικές εταιρείες μικρής κεφαλαιοποίησης, αναλύοντας ακατέργαστα λογιστικά δεδομένα XBRL σε συνδυασμό με τιμές αγοράς σε πραγματικό χρόνο.

Αρχιτεκτονική Συστήματος
Εξαγωγή Δεδομένων (edgar_client.py, data_extractor.py)

Παρακάμπτει τους εμπορικούς παρόχους, αντλώντας ακατέργαστα JSON αρχεία (CompanyFacts) απευθείας από τη SEC.

Επιβάλλει αυστηρή συμμόρφωση: όριο ρυθμού κλήσεων 0.15s, exponential backoff, και υποχρεωτική δήλωση User-Agent.

Χρησιμοποιεί τοπική προσωρινή μνήμη (disk caching) για την αποτροπή εξάντλησης RAM και την ελαχιστοποίηση του χρόνου εκτέλεσης σε σαρώσεις μεγάλων δεικτών.

Κανονικοποίηση XBRL (financial_normalizer.py)

Χαρτογραφεί τις ανομοιογενείς ετικέτες US-GAAP XBRL σε ένα δομημένο Pandas DataFrame.

Εφαρμόζει μηχανισμούς διόρθωσης λογιστικών προσήμων και ιεραρχικών εναλλακτικών ετικετών (fallback arrays) για την αντιμετώπιση κενών στα 10-K.

Ποσοτική Μηχανή (metrics_calculator.py)

Υπολογίζει 8 σύνθετους χρηματοοικονομικούς δείκτες (Net Debt/EBITDA, FCF Yield, Sloan Accruals, Cash Conversion Ratio).

Ενσωματώνει το yfinance για την εισαγωγή ζωντανών τιμών μετοχών και τον υπολογισμό της τρέχουσας Κεφαλαιοποίησης και του Enterprise Value.

Δυναμική Κλαδική Δρομολόγηση (screener_filters.py)

Προσαρμόζει δυναμικά τα μοντέλα πτώχευσης βάσει του κωδικού SIC της κάθε εταιρείας.

Εφαρμόζει το Κλασικό Altman Z-Score για τις βιομηχανίες.

Εφαρμόζει το Altman Z''-Score για τις εταιρείες παροχής υπηρεσιών και λογισμικού, εξαλείφοντας τη μαθηματική στρέβλωση του δείκτη κυκλοφοριακής ταχύτητας παγίων.


### Διαδικασία Ενημέρωσης στο GitHub
Αποθήκευσε το `README.md` στον υπολογιστή σου και εκτέλεσε τις παρακάτω εντολές στο PowerShell για να ενημερώσεις το αποθετήριο:

```powershell
git add README.md
git commit -m "docs: Added comprehensive bilingual README with architectural details"
git push
