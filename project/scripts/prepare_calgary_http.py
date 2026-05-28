from __future__ import annotations

import argparse
import gzip
import re
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


DATASET_URL = "ftp://ita.ee.lbl.gov/traces/calgary_access_log.gz"
TIMESTAMP_RE = re.compile(r"\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}):\d{2}:\d{2} [+-]\d{4}\]")


def download_raw(raw_path: Path, url: str = DATASET_URL) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return

    with urllib.request.urlopen(url) as response:
        raw_path.write_bytes(response.read())


def build_hourly_requests(raw_path: Path) -> pd.DataFrame:
    counts: Counter[datetime] = Counter()
    total_lines = 0
    parsed_lines = 0

    with gzip.open(raw_path, "rt", encoding="latin1") as file:
        for line in file:
            total_lines += 1
            match = TIMESTAMP_RE.search(line)
            if not match:
                continue

            timestamp_hour = datetime.strptime(match.group(1), "%d/%b/%Y:%H")
            counts[timestamp_hour] += 1
            parsed_lines += 1

    if not counts:
        raise ValueError(f"No timestamps parsed from {raw_path}")

    hourly_index = pd.date_range(min(counts), max(counts), freq="h")
    hourly_df = pd.DataFrame(
        {
            "timestamp": hourly_index,
            "requests": [counts.get(ts.to_pydatetime(), 0) for ts in hourly_index],
        }
    )

    hourly_df.attrs["total_lines"] = total_lines
    hourly_df.attrs["parsed_lines"] = parsed_lines
    return hourly_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Calgary-HTTP hourly request counts.")
    parser.add_argument("--raw-path", default="data/raw/calgary_access_log.gz")
    parser.add_argument("--output-path", default="data/processed/calgary_http_hourly.csv")
    parser.add_argument("--download", action="store_true", help="Download raw file if it is missing.")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    output_path = Path(args.output_path)

    if args.download:
        download_raw(raw_path)

    hourly_df = build_hourly_requests(raw_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output_path, index=False)

    print(f"Raw log: {raw_path}")
    print(f"Hourly CSV: {output_path}")
    print(f"Parsed lines: {hourly_df.attrs['parsed_lines']:,} / {hourly_df.attrs['total_lines']:,}")
    print(f"Rows: {len(hourly_df):,}")
    print(f"Period: {hourly_df['timestamp'].iloc[0]} -> {hourly_df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
