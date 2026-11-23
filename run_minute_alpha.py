"""Convenience entry point for the Massive minute-bar research pipeline."""

from minute_alpha import parse_args, run_pipeline


if __name__ == "__main__":
    run_pipeline(parse_args())
