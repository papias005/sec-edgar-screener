"""
screener_filters.py
-------------------
Hard-exclusion filter functions for the QuantitativeScreener.

Design contract
---------------
Each public filter function in this module has the signature:

    def filter_*(df: pd.DataFrame) -> pd.Series:
        \"\"\"Return a boolean Series (True if the row PASSES, False if it should be EXCLUDED).\"\"\"

NaN policy (CRITICAL)
---------------------
Any NaN in a metric that the filter depends on triggers an automatic
EXCLUSION (returns False).  Data is never imputed.

SIC-based sector exclusions
---------------------------
Financials:  SIC 6000–6799
Mining:      SIC 1000–1499
Biotech:     SIC 2836 (pharmaceutical preparations) and
             SIC 8731 (commercial physical & biological research)

Note: SIC codes are integers. None / NaN SIC → excluded (unknown sector).

Quantitative thresholds (hard limits)
--------------------------------------
Manufacturing Altman Z-Score > 2.99
Non-Manufacturing Altman Z''-Score > 2.60
Net Debt / EBITDA    < 3.0    (leverage ceiling)
Current Ratio        > 1.5    (short-term liquidity floor)
Sloan Accruals Ratio < 0.10   (earnings quality, Sloan 1996)
CCR (CFO/EBITDA)     > 1.0    (cash conversion floor)
EV/EBITDA            > 0      (positive enterprise value only)
FCF Yield            > 0      (requires positive free cash flow)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SIC code ranges for excluded sectors
# ---------------------------------------------------------------------------

# Financials: banks, insurance, REITs, brokerage, mortgage companies
_SIC_FINANCIALS_START = 6000
_SIC_FINANCIALS_END = 6799

# Mining: metal, coal, oil & gas upstream, minerals
_SIC_MINING_START = 1000
_SIC_MINING_END = 1499

# Clinical biotech / pharma research (keep established pharma; cut pure R&D)
_SIC_BIOTECH_CODES: frozenset[int] = frozenset({
    2836,   # Pharmaceutical preparations (speculative biotech)
    8731,   # Commercial physical & biological research (pre-revenue biotech)
})

# ---------------------------------------------------------------------------
# Filter 1: Sector Exclusion (SIC code)
# ---------------------------------------------------------------------------

def filter_sector_exclusion(df: pd.DataFrame) -> pd.Series:
    """
    Exclude Financials, Mining, and clinical Biotech by SIC code.

    A missing (NaN / None) SIC code triggers exclusion — we never assume
    a sector for an unclassified company.
    """
    sic = pd.to_numeric(df["sic"], errors="coerce")
    
    is_fin = sic.between(_SIC_FINANCIALS_START, _SIC_FINANCIALS_END)
    is_min = sic.between(_SIC_MINING_START, _SIC_MINING_END)
    is_bio = sic.isin(_SIC_BIOTECH_CODES)
    
    return ~(is_fin | is_min | is_bio | sic.isna())


# ---------------------------------------------------------------------------
# Filter 2: Altman Z-Score
# ---------------------------------------------------------------------------

def filter_altman_z(df: pd.DataFrame) -> pd.Series:
    """
    Altman Z-Score MUST be strictly > 2.99 (Manufacturing) or > 2.60 (Non-Manufacturing).

    A NaN Z-Score is an automatic exclusion — it means foundational balance
    sheet data was missing and the solvency cannot be assessed.
    """
    sic = pd.to_numeric(df["sic"], errors="coerce")
    z = pd.to_numeric(df["altman_z"], errors="coerce")
    
    is_manufacturing = sic.between(2000, 3999)
    
    manufacturing_pass = is_manufacturing & (z > 2.99)
    other_pass = (~is_manufacturing) & sic.notna() & (z > 2.60)
    
    return (manufacturing_pass | other_pass) & z.notna()


# ---------------------------------------------------------------------------
# Filter 3: Net Debt / EBITDA
# ---------------------------------------------------------------------------

def filter_net_debt_ebitda(df: pd.DataFrame) -> pd.Series:
    """
    Net Debt / EBITDA MUST be < 3.0.

    NaN → excluded.
    Negative net debt (net cash position) always passes (< 3.0).
    """
    ratio = pd.to_numeric(df["net_debt_ebitda"], errors="coerce")
    return ratio < 3.0


# ---------------------------------------------------------------------------
# Filter 4: Current Ratio
# ---------------------------------------------------------------------------

def filter_current_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Current Ratio MUST be > 1.5.

    NaN → excluded.
    """
    cr = pd.to_numeric(df["current_ratio"], errors="coerce")
    return cr > 1.5


# ---------------------------------------------------------------------------
# Filter 5: Sloan Accruals Ratio
# ---------------------------------------------------------------------------

def filter_sloan_accruals(df: pd.DataFrame) -> pd.Series:
    """
    Sloan Accruals Ratio MUST be < 0.10.

    Sloan (1996): high accruals relative to assets are predictive of lower
    future earnings — a sign of aggressive revenue recognition.

    NaN → excluded.
    """
    sloan = pd.to_numeric(df["sloan_accruals"], errors="coerce")
    return sloan < 0.10


# ---------------------------------------------------------------------------
# Filter 6: Cash Conversion Ratio (CCR = CFO / EBITDA)
# ---------------------------------------------------------------------------

def filter_ccr(df: pd.DataFrame) -> pd.Series:
    """
    Cash Conversion Ratio (CFO / EBITDA) MUST be > 1.0.

    A CCR > 1 means the company converts its EBITDA to actual cash at a rate
    exceeding 100%, confirming the quality of reported earnings.

    NaN → excluded.
    """
    ccr = pd.to_numeric(df["ccr"], errors="coerce")
    return ccr > 1.0


# ---------------------------------------------------------------------------
# Filter 7: EV/EBITDA > 0 (positive enterprise value, profitability gate)
# ---------------------------------------------------------------------------

def filter_ev_ebitda_positive(df: pd.DataFrame) -> pd.Series:
    """
    EV/EBITDA MUST be > 0.

    Excludes companies with negative EBITDA (unprofitable) or negative
    enterprise value (rare, usually signals data problems).

    NaN → excluded (market cap likely missing → yfinance failed).
    """
    ev_ebitda = pd.to_numeric(df["ev_ebitda"], errors="coerce")
    return ev_ebitda > 0.0


# ---------------------------------------------------------------------------
# Filter 8: FCF Yield > 0
# ---------------------------------------------------------------------------

def filter_fcf_yield_positive(df: pd.DataFrame) -> pd.Series:
    """
    Free Cash Flow Yield (FCF / Market Cap) MUST be > 0.

    Excludes companies that do not generate positive free cash flow.
    NaN → excluded (either FCF or market cap is missing).
    """
    fcf = pd.to_numeric(df["fcf_yield"], errors="coerce")
    return fcf > 0.0


# ---------------------------------------------------------------------------
# Convenience: the ordered list of all DataFrame-level hard filters
# ---------------------------------------------------------------------------

ALL_HARD_FILTERS: list[tuple[str, callable]] = [
    ("sector_exclusion",    filter_sector_exclusion),
    ("altman_z",            filter_altman_z),
    ("net_debt_ebitda",     filter_net_debt_ebitda),
    ("current_ratio",       filter_current_ratio),
    ("sloan_accruals",      filter_sloan_accruals),
    ("ccr",                 filter_ccr),
    ("ev_ebitda_positive",  filter_ev_ebitda_positive),
    ("fcf_yield_positive",  filter_fcf_yield_positive),
]
"""
Ordered list of (name, filter_fn) tuples.

Import and pass directly into QuantitativeScreener.screen() if you want the
full default filter set, or cherry-pick individual functions for custom runs.
"""
