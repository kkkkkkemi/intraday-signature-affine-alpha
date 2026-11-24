"""Research universes and their GICS sectors.

SP100_AT_START is the S&P 100 as it stood at the start of the downloadable sample
(late 2024), reconstructed from the September 21, 2026 constituent list by undoing
every later index change announced by S&P Dow Jones Indices:

* Mar 24, 2025  added PLTR, ISRG, NOW;          deleted DOW, KHC, F
* Sep 22, 2025  added UBER;                     deleted CHTR
* Mar 23, 2026  added MU, LRCX, AMAT, GEV;      deleted PYPL, AIG, MET, TGT
* Jun 29/30, 2026 Honeywell Aerospace (HONA) replaced Honeywell International (HON)
* Sep 21, 2026  added DELL, PANW, ANET, SNDK;   deleted HONA, NKE, SPG, CL

The September and December 2024 rebalances made no S&P 100 changes, so the list
also holds for a sample starting between September and December 2024.  Using the
start-of-sample membership (not today's) avoids selecting names because they later
grew into the index.  GOOG is dropped because GOOGL is the same company.

Sectors are GICS sectors after the March 2023 reclassification, under which Visa,
Mastercard and PayPal are Financials (the original 25-name map wrongly kept V and
MA in Information Technology).
"""

from __future__ import annotations

MEGA25 = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
    "AMD", "INTC", "ORCL", "CRM", "NFLX", "TSLA",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "WMT", "COST", "HD", "JNJ", "UNH", "XOM",
)

SECTOR_BY_SYMBOL: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "ACN": "Information Technology", "ADBE": "Information Technology",
    "AMD": "Information Technology", "AVGO": "Information Technology", "CRM": "Information Technology",
    "CSCO": "Information Technology", "IBM": "Information Technology", "INTC": "Information Technology",
    "INTU": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "ORCL": "Information Technology", "QCOM": "Information Technology", "TXN": "Information Technology",
    # Communication Services
    "CHTR": "Communication Services", "CMCSA": "Communication Services", "DIS": "Communication Services",
    "GOOGL": "Communication Services", "META": "Communication Services", "NFLX": "Communication Services",
    "T": "Communication Services", "TMUS": "Communication Services", "VZ": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "BKNG": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "HD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # Consumer Staples
    "CL": "Consumer Staples", "COST": "Consumer Staples", "KHC": "Consumer Staples", "KO": "Consumer Staples",
    "MDLZ": "Consumer Staples", "MO": "Consumer Staples", "PEP": "Consumer Staples", "PG": "Consumer Staples",
    "PM": "Consumer Staples", "TGT": "Consumer Staples", "WMT": "Consumer Staples",
    # Financials
    "AIG": "Financials", "AXP": "Financials", "BAC": "Financials", "BLK": "Financials", "BNY": "Financials",
    "BRK.B": "Financials", "C": "Financials", "COF": "Financials", "GS": "Financials", "JPM": "Financials",
    "MA": "Financials", "MET": "Financials", "MS": "Financials", "PYPL": "Financials", "SCHW": "Financials",
    "USB": "Financials", "V": "Financials", "WFC": "Financials",
    # Health Care
    "ABBV": "Health Care", "ABT": "Health Care", "AMGN": "Health Care", "BMY": "Health Care",
    "CVS": "Health Care", "DHR": "Health Care", "GILD": "Health Care", "JNJ": "Health Care",
    "LLY": "Health Care", "MDT": "Health Care", "MRK": "Health Care", "PFE": "Health Care",
    "TMO": "Health Care", "UNH": "Health Care",
    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials", "EMR": "Industrials", "FDX": "Industrials",
    "GD": "Industrials", "GE": "Industrials", "HON": "Industrials", "LMT": "Industrials", "MMM": "Industrials",
    "RTX": "Industrials", "UNP": "Industrials", "UPS": "Industrials",
    # Energy
    "COP": "Energy", "CVX": "Energy", "XOM": "Energy",
    # Utilities
    "DUK": "Utilities", "NEE": "Utilities", "SO": "Utilities",
    # Materials
    "DOW": "Materials", "LIN": "Materials",
    # Real Estate
    "AMT": "Real Estate", "SPG": "Real Estate",
}

SP100_AT_START = tuple(sorted(SECTOR_BY_SYMBOL))

# Exchange ticker history for names whose symbol changed inside the sample.  The
# data vendor is queried by symbol string, so each segment is downloaded under
# the symbol valid at the time and stored under the current symbol.
# BNY Mellon traded as BK through 2026-05-20 and as BNY from 2026-05-21; before
# that date "BNY" belonged to an unrelated, thinly traded security.
SYMBOL_HISTORY: dict[str, tuple[tuple[str, str | None, str | None], ...]] = {
    "BNY": (("BK", None, "2026-05-20"), ("BNY", "2026-05-21", None)),
}

UNIVERSES = {"mega25": MEGA25, "sp100": SP100_AT_START}

# Market and SPDR sector ETFs used as context features (downloaded to DATA_DIR/etf).
MARKET_ETF = "SPY"
SECTOR_ETF: dict[str, str] = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
ETF_TICKERS = (MARKET_ETF, *sorted(set(SECTOR_ETF.values())))


def universe(name: str) -> tuple[str, ...]:
    if name not in UNIVERSES:
        raise ValueError(f"Unknown universe {name!r}; choose one of {sorted(UNIVERSES)}.")
    return UNIVERSES[name]


if __name__ == "__main__":
    from collections import Counter

    print(f"S&P 100 at sample start: {len(SP100_AT_START)} names")
    for sector, count in sorted(Counter(SECTOR_BY_SYMBOL.values()).items(), key=lambda item: -item[1]):
        print(f"  {sector:<24} {count}")
