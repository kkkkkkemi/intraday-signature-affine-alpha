"""Check minute-bar coverage of the configured universe before training.

Run directly after download_data.py.  It reads every ticker file in DATA_DIR and
reports, without changing anything:

* tickers with no file, and each ticker's first/last date and session count;
* sessions in which a ticker has no bars at all.  The training grid needs every
  ticker in every window, so such a session is lost for *all* tickers;
* the share of regular-session minutes with a real bar (illiquid names have more
  carried-forward minutes, flagged by has_trade/staleness in the features);
* a greedy suggestion of which tickers to drop to recover lost sessions.

Output: <DATA_DIR>/coverage_report.json and coverage_by_ticker.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_settings import DATA_DIR, TICKERS, USE_ETF_CONTEXT
from universe import ETF_TICKERS

PROJECT_DIR = Path(__file__).resolve().parent
FULL_SESSION_MINUTES = 390
# A ticker missing more than this share of sessions is suggested for removal.
MAX_MISSING_SESSION_SHARE = 0.02


def session_minutes(frame: pd.DataFrame) -> pd.Series:
    """Number of regular-session bars per New York trading date."""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York")
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    regular = (minutes >= 570) & (minutes < 960) & (timestamps.dt.dayofweek < 5)
    return timestamps[regular].dt.date.astype(str).value_counts().sort_index()


def session_last_minute(frame: pd.DataFrame) -> pd.Series:
    """Minute of day (New York) of each session's last bar."""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York")
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    regular = (minutes >= 570) & (minutes < 960) & (timestamps.dt.dayofweek < 5)
    return minutes[regular].groupby(timestamps[regular].dt.date.astype(str)).max()


def analyse(bars_by_ticker: dict[str, pd.DataFrame], tickers: list[str]) -> tuple[dict, pd.DataFrame]:
    if not bars_by_ticker:
        empty = {
            "tickers_requested": len(tickers), "tickers_with_files": 0, "missing_files": sorted(tickers),
            "calendar_sessions": 0, "calendar_range": [None, None], "sessions_lost_with_all_tickers": 0,
            "lost_sessions_examples": [], "suggested_drops": [], "sessions_lost_after_drops": 0,
            "lowest_bar_coverage": [], "short_sessions": {}, "final_session": None, "data_dir": str(DATA_DIR),
        }
        return empty, pd.DataFrame()
    per_ticker = {ticker: session_minutes(frame) for ticker, frame in bars_by_ticker.items()}
    last_minute = (
        pd.concat({ticker: session_last_minute(frame) for ticker, frame in bars_by_ticker.items()}, axis=1).max(axis=1)
        if bars_by_ticker
        else pd.Series(dtype=float)
    )
    missing_files = sorted(set(tickers) - set(per_ticker))
    # Calendar: sessions where at least half of the available tickers traded.
    counts = pd.concat(per_ticker, axis=1).fillna(0.0) if per_ticker else pd.DataFrame()
    present = counts.gt(0)
    calendar = counts.index[present.sum(axis=1) >= 0.5 * max(len(per_ticker), 1)]
    counts = counts.reindex(calendar).fillna(0.0)
    present = counts.gt(0)

    # Early closes have fewer minutes for everyone; measure against the session's max.
    session_length = counts.max(axis=1).replace(0, np.nan)
    rows = []
    for ticker in sorted(per_ticker):
        column = counts[ticker]
        missing_sessions = list(calendar[~present[ticker].to_numpy()])
        traded = column[column > 0]
        rows.append(
            {
                "ticker": ticker,
                "first_session": traded.index.min() if len(traded) else None,
                "last_session": traded.index.max() if len(traded) else None,
                "sessions_present": int(present[ticker].sum()),
                "sessions_missing": len(missing_sessions),
                "missing_share": len(missing_sessions) / max(len(calendar), 1),
                "median_bar_coverage": float((column / session_length)[column > 0].median()) if len(traded) else 0.0,
                "missing_session_examples": ", ".join(missing_sessions[:5]),
            }
        )
    table = pd.DataFrame.from_records(rows).sort_values(["sessions_missing", "median_bar_coverage"], ascending=[False, True])

    lost = ~present.all(axis=1)
    # Greedy removal: drop the ticker responsible for most lost sessions until the
    # remaining ones are each missing at most MAX_MISSING_SESSION_SHARE of sessions.
    suggestion: list[str] = []
    remaining = present.copy()
    while len(remaining.columns) > 3:
        missing_counts = (~remaining).sum(axis=0)
        worst = missing_counts.idxmax()
        if missing_counts[worst] <= MAX_MISSING_SESSION_SHARE * len(calendar) or missing_counts[worst] == 0:
            break
        suggestion.append(str(worst))
        remaining = remaining.drop(columns=worst)
    lost_after = ~remaining.all(axis=1)

    last_minute = last_minute.reindex(calendar)
    short = last_minute[last_minute < 15 * 60 + 59]
    final_session = str(calendar.max()) if len(calendar) else None
    report = {
        "data_dir": str(DATA_DIR),
        "short_sessions": {str(day): f"{int(minute) // 60:02d}:{int(minute) % 60:02d}" for day, minute in short.items()},
        "note_short_sessions": (
            "Sessions ending before 12:59 NY, and a final session ending before 15:59, are dropped "
            "automatically by training (minute_alpha.drop_incomplete_sessions); 13:00 early closes are kept."
        ),
        "final_session": final_session,
        "tickers_requested": len(tickers),
        "tickers_with_files": len(per_ticker),
        "missing_files": missing_files,
        "calendar_sessions": int(len(calendar)),
        "calendar_range": [str(calendar.min()) if len(calendar) else None, str(calendar.max()) if len(calendar) else None],
        "sessions_lost_with_all_tickers": int(lost.sum()),
        "lost_sessions_examples": [str(day) for day in calendar[lost.to_numpy()][:10]],
        "suggested_drops": suggestion,
        "sessions_lost_after_drops": int(lost_after.sum()),
        "lowest_bar_coverage": table.nsmallest(5, "median_bar_coverage")[["ticker", "median_bar_coverage"]]
        .round(3)
        .to_dict(orient="records"),
    }
    return report, table


def main() -> None:
    data_dir = Path(DATA_DIR)
    data_dir = data_dir if data_dir.is_absolute() else PROJECT_DIR / data_dir
    bars = {}
    for ticker in TICKERS:
        path = data_dir / f"{ticker}.parquet"
        if path.exists():
            bars[ticker] = pd.read_parquet(path, columns=["timestamp"])
    report, table = analyse(bars, list(TICKERS))
    if USE_ETF_CONTEXT:
        etf_dir = data_dir / "etf"
        etf_bars = {
            ticker: pd.read_parquet(etf_dir / f"{ticker}.parquet", columns=["timestamp"])
            for ticker in ETF_TICKERS
            if (etf_dir / f"{ticker}.parquet").exists()
        }
        etf_report, _ = analyse(etf_bars, list(ETF_TICKERS))
        report["etf"] = {
            key: etf_report[key]
            for key in ("tickers_with_files", "missing_files", "calendar_range", "sessions_lost_with_all_tickers", "lowest_bar_coverage")
        }
    (data_dir / "coverage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    table.to_csv(data_dir / "coverage_by_ticker.csv", index=False)

    print(f"Universe: {report['tickers_requested']} requested, {report['tickers_with_files']} files found")
    if report["missing_files"]:
        print(f"  Missing files (download again or drop): {report['missing_files']}")
    print(f"Calendar: {report['calendar_sessions']} sessions {report['calendar_range']}")
    if "etf" in report:
        etf = report["etf"]
        print(
            f"Context ETFs: {etf['tickers_with_files']}/{len(ETF_TICKERS)} files, missing {etf['missing_files'] or 'none'}, "
            f"range {etf['calendar_range']}, sessions with a missing ETF {etf['sessions_lost_with_all_tickers']}"
        )
    if report["short_sessions"]:
        print(f"Sessions ending before 15:59 NY (last bar time): {report['short_sessions']}")
        print("  -> 13:00 early closes are kept; partial days (incl. an unfinished final day) are dropped in training.")
    print(f"Sessions lost because some ticker has no bars that day: {report['sessions_lost_with_all_tickers']}")
    if report["lost_sessions_examples"]:
        print(f"  e.g. {report['lost_sessions_examples']}")
    print(f"Suggested drops: {report['suggested_drops'] or 'none'} -> lost sessions after drops: {report['sessions_lost_after_drops']}")
    print("Tickers with missing sessions or lowest bar coverage:")
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(table.head(12).to_string(index=False))
    print(f"\nWritten to {data_dir / 'coverage_report.json'} and coverage_by_ticker.csv")


if __name__ == "__main__":
    main()
