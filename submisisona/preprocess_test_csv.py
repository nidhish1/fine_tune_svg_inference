#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def normalize_prompt(prompt: str) -> str:
    # Keep content unchanged except for surrounding whitespace.
    return (prompt or "").strip()


def preprocess_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    processed: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    dropped_empty_prompt = 0
    dropped_duplicate_id = 0
    generated_ids = 0

    for i, row in enumerate(rows):
        raw_id = (row.get("sample_id") or row.get("id") or "").strip()
        prompt = normalize_prompt(row.get("prompt", ""))

        if not prompt:
            dropped_empty_prompt += 1
            continue

        if not raw_id:
            raw_id = f"row_{i:06d}"
            generated_ids += 1

        if raw_id in seen_ids:
            dropped_duplicate_id += 1
            continue

        seen_ids.add(raw_id)
        processed.append({"sample_id": raw_id, "prompt": prompt})

    stats = {
        "input_rows": len(rows),
        "output_rows": len(processed),
        "dropped_empty_prompt": dropped_empty_prompt,
        "dropped_duplicate_id": dropped_duplicate_id,
        "generated_ids": generated_ids,
    }
    return processed, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy and preprocess test CSV for final inference.")
    parser.add_argument("--input-csv", required=True, help="Path to original test.csv")
    parser.add_argument("--copy-csv", required=True, help="Path to copied raw test.csv")
    parser.add_argument("--output-csv", required=True, help="Path to processed CSV")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    copy_csv = Path(args.copy_csv)
    output_csv = Path(args.output_csv)

    copy_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_csv, copy_csv)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    processed, stats = preprocess_rows(rows)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prompt"])
        writer.writeheader()
        writer.writerows(processed)

    print(f"input_csv: {input_csv}")
    print(f"copied_csv: {copy_csv}")
    print(f"processed_csv: {output_csv}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
