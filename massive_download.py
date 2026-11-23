"""Download Massive/Polygon minute aggregates for the intraday-alpha pipeline.

The free Massive Stocks Basic plan supplies two years of historical minute aggregates.
This downloader deliberately uses only that product: it does not claim to download
quotes, trades, or order-book depth.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


API_URL = "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
DEFAULT_TICKERS = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
    "AMD", "INTC", "ORCL", "CRM", "NFLX", "TSLA",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "WMT", "COST", "HD", "JNJ", "UNH", "XOM",
)


class MassiveDownloadError(RuntimeError):
    """Raised when Massive rejects or cannot fulfil a data request."""


def load_local_env(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE entries without adding a python-dotenv dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _rth_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep regular US-equity trading hours without fabricating missing bars."""
    ts_ny = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York")
    is_weekday = ts_ny.dt.dayofweek < 5
    minutes = ts_ny.dt.hour * 60 + ts_ny.dt.minute
    is_rth = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    return frame.loc[is_weekday & is_rth].copy()


def _normalise_results(results: list[dict], ticker: str) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "transactions"]
        )

    frame = pd.DataFrame(results).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "transactions"}
    )
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise MassiveDownloadError(f"{ticker}: aggregate response is missing columns {missing}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["symbol"] = ticker
    for col in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if col not in frame:
            frame[col] = pd.NA
    frame = frame[["timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap", "transactions"]]
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return _rth_only(frame).reset_index(drop=True)


class MassiveClient:
    """Minimal paginated client that honours the Basic plan's 5 calls/minute limit."""

    def __init__(self, api_key: str, calls_per_minute: int = 5) -> None:
        if not api_key:
            raise ValueError("Set MASSIVE_API_KEY before downloading data.")
        self.api_key = api_key
        self.min_interval_seconds = 60.0 / calls_per_minute
        self._last_request_at = 0.0
        self.session = requests.Session()

    def _get(self, url: str, params: dict | None = None, max_attempts: int = 6) -> dict:
        """GET with pacing, plus retry on 429, 5xx and network errors (exponential backoff)."""
        request_params = {"apiKey": self.api_key}
        if params:
            request_params.update(params)
        for attempt in range(1, max_attempts + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            try:
                response = self.session.get(url, params=request_params, timeout=60)
            except requests.RequestException as exc:
                self._last_request_at = time.monotonic()
                if attempt == max_attempts:
                    raise MassiveDownloadError(f"Network error after {attempt} attempts: {exc}") from exc
                wait = min(15.0 * 2 ** (attempt - 1), 300.0)
                print(f"    network error ({exc.__class__.__name__}); retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            self._last_request_at = time.monotonic()
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == max_attempts:
                    raise MassiveDownloadError(
                        f"Massive returned {response.status_code} after {attempt} attempts; rerun later, "
                        "completed files are retained."
                    )
                wait = 65.0 if response.status_code == 429 else min(15.0 * 2 ** (attempt - 1), 300.0)
                print(f"    HTTP {response.status_code}; retrying in {wait:.0f}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
                continue
            break
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise MassiveDownloadError(f"Massive request failed ({response.status_code}): {response.text[:500]}") from exc
        payload = response.json()
        if payload.get("status") == "ERROR":
            raise MassiveDownloadError(f"Massive returned an error: {payload.get('error', payload)}")
        return payload

    def minute_aggregates(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        url = API_URL.format(ticker=ticker.upper(), start=start.isoformat(), end=end.isoformat())
        payload = self._get(url, {"adjusted": "true", "sort": "asc", "limit": 50_000})
        results = list(payload.get("results", []))
        next_url = payload.get("next_url")
        while next_url:
            payload = self._get(next_url)
            results.extend(payload.get("results", []))
            next_url = payload.get("next_url")
        return _normalise_results(results, ticker.upper())


def download_tickers(
    tickers: Iterable[str],
    output_dir: Path,
    start: date,
    end: date,
    overwrite: bool = False,
    api_key: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    load_local_env()
    client = MassiveClient(api_key or os.environ.get("MASSIVE_API_KEY", ""))
    manifest_path = output_dir / "manifest.json"
    previous_files: dict[str, object] = {}
    if manifest_path.exists():
        try:
            previous_files = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {}))
        except (json.JSONDecodeError, OSError):
            previous_files = {}
    manifest: dict[str, object] = {
        "source": "Massive Stocks minute aggregates",
        "endpoint": "v2/aggs/ticker/{ticker}/range/1/minute/{from}/{to}",
        "adjusted": True,
        "regular_trading_hours_only": True,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "files": previous_files,
    }
    failed: list[str] = []
    empty: list[str] = []

    for ticker in sorted({item.upper() for item in tickers}):
        destination = output_dir / f"{ticker}.parquet"
        if destination.exists() and not overwrite:
            print(f"Skipping {ticker}: {destination.name} already exists (use --overwrite to refresh).")
            continue
        print(f"Downloading {ticker}: {start} to {end} …")
        try:
            frame = client.minute_aggregates(ticker, start, end)
        except MassiveDownloadError as exc:
            print(f"  FAILED {ticker}: {exc}")
            failed.append(ticker)
            continue
        if frame.empty:
            print(f"  No regular-session minute aggregates returned for {ticker}.")
            empty.append(ticker)
            continue
        frame.to_parquet(destination, index=False)
        manifest["files"][ticker] = {
            "path": destination.name,
            "rows": int(len(frame)),
            "first_timestamp": frame["timestamp"].min().isoformat(),
            "last_timestamp": frame["timestamp"].max().isoformat(),
        }
        print(f"  Saved {len(frame):,} bars to {destination}")
        # Persist after every ticker so an interrupted run keeps an accurate manifest.
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failed or empty:
        print(f"\nNot downloaded - failed: {failed or 'none'}; empty: {empty or 'none'}. Rerun to retry failures.")


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Download Massive minute OHLCV data for this project.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS), help="US ticker symbols to download")
    parser.add_argument("--start", default=(today - timedelta(days=730)).isoformat(), help="Inclusive ISO start date")
    parser.add_argument("--end", default=today.isoformat(), help="Inclusive ISO end date")
    parser.add_argument("--output-dir", default="minute_data", help="Destination directory for Parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing ticker files")
    parser.add_argument("--api-key", help="Optional Massive API key; .env or MASSIVE_API_KEY is safer for repeated use")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_tickers(
        tickers=args.tickers,
        output_dir=Path(args.output_dir),
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        overwrite=args.overwrite,
        api_key=args.api_key,
    )
