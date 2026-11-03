"""Download Massive minute bars using the Python environment that launches this file.

Put MASSIVE_API_KEY=... in a local .env file first.  Then run this file directly
from your usual Python IDE or double-click it in Explorer; no PowerShell command is
needed.  Edit project_settings.py to change tickers and dates.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from massive_download import download_tickers, load_local_env
from project_settings import DATA_DIR, END_DATE, START_DATE, TICKERS


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    load_local_env(PROJECT_DIR / ".env")
    data_dir = Path(DATA_DIR)
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    download_tickers(
        tickers=TICKERS,
        output_dir=data_dir,
        start=date.fromisoformat(START_DATE),
        end=date.fromisoformat(END_DATE),
    )


if __name__ == "__main__":
    main()
