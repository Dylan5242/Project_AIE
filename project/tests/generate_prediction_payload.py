from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path


DEFAULT_LAST_TIMESTAMP = "1995-10-11T14:00:00"


def parse_positions(raw_value: str | None) -> list[int]:
    """Parse comma-separated 1-based positions."""
    if raw_value is None or raw_value.strip() == "":
        return []
    return [int(item.strip()) for item in raw_value.split(",") if item.strip()]


def validate_positions(
    n: int,
    min_count: int,
    max_count: int,
    min_positions: list[int],
    max_positions: list[int],
) -> None:
    if n < 3:
        raise ValueError("n must be at least 3 because local extrema need left and right neighbors.")

    if len(min_positions) != min_count:
        raise ValueError(f"Expected {min_count} min positions, got {len(min_positions)}.")

    if len(max_positions) != max_count:
        raise ValueError(f"Expected {max_count} max positions, got {len(max_positions)}.")

    all_positions = min_positions + max_positions
    if len(all_positions) != len(set(all_positions)):
        raise ValueError("Min and max positions must not overlap.")

    invalid = [position for position in all_positions if position <= 1 or position >= n]
    if invalid:
        raise ValueError(
            "Local extrema cannot be first or last elements. "
            f"Invalid 1-based positions: {invalid}."
        )


def generate_values(
    n: int,
    min_positions: list[int],
    max_positions: list[int],
    seed: int | None = None,
    low: int = 80,
    high: int = 420,
    margin: int = 35,
) -> list[int]:
    """Generate random request values with forced local minima and maxima."""
    rng = random.Random(seed)
    values = [rng.randint(low, high) for _ in range(n)]

    min_indexes = [position - 1 for position in min_positions]
    max_indexes = [position - 1 for position in max_positions]

    for index in min_indexes:
        left = values[index - 1]
        right = values[index + 1]
        values[index] = max(0, min(left, right) - rng.randint(margin, margin * 3))

    for index in max_indexes:
        left = values[index - 1]
        right = values[index + 1]
        values[index] = max(left, right) + rng.randint(margin, margin * 3)

    return values


def build_payload(values: list[int], last_timestamp: str) -> dict[str, object]:
    datetime.fromisoformat(last_timestamp)
    return {
        "last_timestamp": last_timestamp,
        "requests": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a random /predict request payload with forced local extrema."
    )
    parser.add_argument("--n", type=int, required=True, help="Number of generated request values.")
    parser.add_argument("--min-count", type=int, required=True, help="Number of local minima.")
    parser.add_argument("--max-count", type=int, required=True, help="Number of local maxima.")
    parser.add_argument(
        "--min-positions",
        type=str,
        default="",
        help="Comma-separated 1-based positions for local minima, for example: 10,35,80.",
    )
    parser.add_argument(
        "--max-positions",
        type=str,
        default="",
        help="Comma-separated 1-based positions for local maxima, for example: 20,55,120.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--last-timestamp",
        type=str,
        default=DEFAULT_LAST_TIMESTAMP,
        help="Last observation timestamp for the API payload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save generated JSON payload.",
    )
    args = parser.parse_args()

    min_positions = parse_positions(args.min_positions)
    max_positions = parse_positions(args.max_positions)
    validate_positions(args.n, args.min_count, args.max_count, min_positions, max_positions)

    values = generate_values(
        n=args.n,
        min_positions=min_positions,
        max_positions=max_positions,
        seed=args.seed,
    )
    payload = build_payload(values, args.last_timestamp)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload_json, encoding="utf-8")
        print(f"Saved payload to {args.output}")
    else:
        print(payload_json)


if __name__ == "__main__":
    main()
