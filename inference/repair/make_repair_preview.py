#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import random
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def main():
    parser = argparse.ArgumentParser(description="Create HTML preview for repaired SVG outputs.")
    parser.add_argument("--input-csv", required=True, help="CSV from repair_svg_outputs.py")
    parser.add_argument("--output-html", required=True, help="Output HTML path")
    parser.add_argument("--samples", type=int, default=30, help="How many rows to display")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--only-valid", action="store_true", help="Show only rows with is_valid_xml=1")
    parser.add_argument("--only-graphics", action="store_true", help="Show only rows with has_graphics=1")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.only_valid:
        rows = [r for r in rows if r.get("is_valid_xml") == "1"]
    if args.only_graphics:
        rows = [r for r in rows if r.get("has_graphics") == "1"]

    random.seed(args.seed)
    if rows:
        rows = random.sample(rows, min(args.samples, len(rows)))

    cards = []
    for i, r in enumerate(rows, 1):
        sample_id = html.escape(r.get("sample_id", ""))
        prompt = html.escape(r.get("prompt", ""))
        status = html.escape(r.get("repair_status", ""))
        valid = "valid" if r.get("is_valid_xml") == "1" else "invalid"
        svg = r.get("svg", "")
        svg_escaped = html.escape(svg)

        cards.append(
            f"""
            <section class="card">
              <div class="meta">
                <div><b>#{i}</b> <code>{sample_id}</code></div>
                <div class="pill {valid}">{valid}</div>
              </div>
              <p><b>Prompt:</b> {prompt}</p>
              <p><b>Repair status:</b> <code>{status}</code></p>
              <div class="render">{svg if svg else "<em>No SVG recovered</em>"}</div>
              <details>
                <summary>Show SVG source</summary>
                <pre>{svg_escaped}</pre>
              </details>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Repaired SVG Preview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f7f9; }}
    h1 {{ margin-bottom: 6px; }}
    .sub {{ color: #555; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    .meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
    .pill {{ border-radius: 999px; padding: 3px 10px; font-size: 12px; }}
    .pill.valid {{ background: #e8f8ec; color: #116329; }}
    .pill.invalid {{ background: #fdecec; color: #8d1e1e; }}
    .render {{ border: 1px dashed #bbb; border-radius: 8px; min-height: 180px; padding: 8px; background: #fff; overflow: auto; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #111; color: #eee; padding: 8px; border-radius: 8px; }}
    code {{ background: #f1f3f5; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Repaired SVG Preview</h1>
  <div class="sub">Rows shown: {len(rows)} | only_valid={args.only_valid} | only_graphics={args.only_graphics}</div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(f"input: {in_path}")
    print(f"output: {out_path}")
    print(f"rows_rendered: {len(rows)}")


if __name__ == "__main__":
    main()
