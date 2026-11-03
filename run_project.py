"""One-command workflow: load a local API key, download Massive bars, then train."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from massive_download import DEFAULT_TICKERS, download_tickers, load_local_env
from minute_alpha import RunConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    today = date.today()
    parser = argparse.ArgumentParser(description="Download Massive minute data and run the signature-affine study.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--start", default=(today - timedelta(days=730)).isoformat())
    parser.add_argument("--end", default=today.isoformat())
    parser.add_argument("--data-dir", default="minute_data")
    parser.add_argument("--output-dir", default="artifacts/signature_affine_n25_sig91_state7_ms3_clv_sector_shrink5_target_sector_mean_normalscore_ens42_z0anchor5scale_pp_pca12_pv_pca12_continuous_z1_gate1bp_m2bp_w30_h30_tr5_ev30")
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--holding-minutes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--sector-median-shrinkage-count", type=float, default=5.0)
    parser.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", help="Use once; prefer MASSIVE_API_KEY in .env for local development")
    parser.add_argument("--skip-download", action="store_true", help="Train only, using existing Parquet files")
    parser.add_argument("--overwrite-data", action="store_true", help="Refresh downloaded ticker files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_local_env()
    if not args.skip_download:
        download_tickers(
            tickers=args.tickers,
            output_dir=Path(args.data_dir),
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            overwrite=args.overwrite_data,
            api_key=args.api_key,
        )
    config = RunConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_minutes=args.window_minutes,
        holding_minutes=args.holding_minutes,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        sector_median_shrinkage_count=args.sector_median_shrinkage_count,
        ensemble_seeds=tuple(args.ensemble_seeds),
        symbols=tuple(args.tickers),
        seed=args.seed,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
