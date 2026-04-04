#!/usr/bin/env python3
import argparse
import csv
import html
import random
from pathlib import Path

from svg_semantic_tokenization import (
    lines_to_tokens,
    semantic_tokens_to_svg,
    svg_to_semantic_tokens,
    tokens_to_lines,
)


def tokenize_and_rebuild_csv(input_csv: Path, tokenized_csv: Path, final_saved_csv: Path) -> dict:
    tokenized_csv.parent.mkdir(parents=True, exist_ok=True)
    final_saved_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    errors = 0

    with input_csv.open("r", newline="", encoding="utf-8") as src, tokenized_csv.open(
        "w", newline="", encoding="utf-8"
    ) as tok_dst, final_saved_csv.open("w", newline="", encoding="utf-8") as final_dst:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing header.")

        token_fields = [f for f in reader.fieldnames if f != "svg"] + ["svg_tokens"]
        final_fields = list(reader.fieldnames)

        tok_writer = csv.DictWriter(tok_dst, fieldnames=token_fields)
        final_writer = csv.DictWriter(final_dst, fieldnames=final_fields)
        tok_writer.writeheader()
        final_writer.writeheader()

        for row in reader:
            total += 1
            original_svg = row.get("svg", "")
            token_lines = ""
            rebuilt_svg = original_svg

            try:
                tokens = svg_to_semantic_tokens(original_svg)
                token_lines = "\n".join(tokens_to_lines(tokens))
                rebuilt_svg = semantic_tokens_to_svg(lines_to_tokens(token_lines.splitlines()))
            except Exception:
                errors += 1
                token_lines = ""
                rebuilt_svg = original_svg

            token_row = {k: v for k, v in row.items() if k != "svg"}
            token_row["svg_tokens"] = token_lines
            tok_writer.writerow(token_row)

            final_row = dict(row)
            final_row["svg"] = rebuilt_svg
            final_writer.writerow(final_row)

    return {"rows": total, "errors": errors}


def build_comparison_preview(
    safe_csv: Path, final_saved_csv: Path, output_html: Path, samples: int, seed: int
) -> dict:
    with safe_csv.open("r", newline="", encoding="utf-8") as f:
        safe_rows = list(csv.DictReader(f))
    with final_saved_csv.open("r", newline="", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))

    final_by_id = {r.get("id"): r for r in final_rows}
    matched = []
    for s in safe_rows:
        rid = s.get("id")
        fr = final_by_id.get(rid)
        if fr:
            matched.append((s, fr))

    rng = random.Random(seed)
    k = min(samples, len(matched))
    sampled = rng.sample(matched, k) if k else []

    cards = []
    for i, (safe_row, final_row) in enumerate(sampled, start=1):
        rid = html.escape(safe_row.get("id", ""))
        prompt = html.escape(safe_row.get("prompt", ""))
        safe_svg = safe_row.get("svg", "")
        final_svg = final_row.get("svg", "")
        cards.append(
            f"""
            <section class="card">
              <div class="meta"><strong>Sample:</strong> {i} &nbsp; <strong>ID:</strong> {rid}<br><strong>Prompt:</strong> {prompt}</div>
              <div class="grid">
                <div class="panel"><h3>Safe Canonicalized</h3><div class="svg-box">{safe_svg}</div></div>
                <div class="panel"><h3>Final Saved (token -> revert)</h3><div class="svg-box">{final_svg}</div></div>
              </div>
            </section>
            """
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Safe vs Final Saved Preview</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f6f8fb; color: #111; }}
    .card {{ background: #fff; border: 1px solid #e4e8ef; border-radius: 12px; padding: 12px; margin-bottom: 12px; }}
    .meta {{ margin-bottom: 10px; line-height: 1.4; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .panel {{ border: 1px solid #e7ebf2; border-radius: 8px; padding: 8px; background: #fcfdff; }}
    .panel h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .svg-box {{ min-height: 220px; display: flex; align-items: center; justify-content: center; background: #fff; border: 1px dashed #cfd7e3; border-radius: 8px; padding: 8px; overflow: hidden; }}
    .svg-box svg {{ width: 200px; height: 200px; max-width: 100%; max-height: 100%; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Safe Canonicalized vs Final Saved</h1>
  <p>{k} random matched samples by <code>id</code>.</p>
  {''.join(cards)}
</body>
</html>
"""
    output_html.write_text(html_doc, encoding="utf-8")
    return {"matched": len(matched), "sampled": k}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SVG tokenization -> reconstruction pipeline.")
    parser.add_argument(
        "--input",
        default="../dl-spring-2026-svg-generation/safe canonicalization/train_safe_canonicalized.csv",
        help="Input safe canonicalized CSV",
    )
    parser.add_argument(
        "--tokenized-output",
        default="tokenized_train_safe.csv",
        help="Output CSV path for semantic tokenized rows",
    )
    parser.add_argument(
        "--final-saved-output",
        default="final_saved_train_safe.csv",
        help="Output CSV path for reconstructed SVG rows",
    )
    parser.add_argument(
        "--preview-output",
        default="preview_safe_vs_final_saved.html",
        help="Output HTML preview path",
    )
    parser.add_argument("--samples", type=int, default=100, help="Preview random sample count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for preview sampling")
    args = parser.parse_args()

    input_csv = Path(args.input)
    tokenized_out = Path(args.tokenized_output)
    final_saved_out = Path(args.final_saved_output)
    preview_out = Path(args.preview_output)

    stats = tokenize_and_rebuild_csv(input_csv, tokenized_out, final_saved_out)
    preview_stats = build_comparison_preview(input_csv, final_saved_out, preview_out, args.samples, args.seed)

    print(f"Tokenized CSV written to: {tokenized_out}")
    print(f"Final saved CSV written to: {final_saved_out}")
    print(f"Preview HTML written to: {preview_out}")
    print(f"Rows processed: {stats['rows']}, tokenization errors: {stats['errors']}")
    print(f"Matched rows: {preview_stats['matched']}, preview sampled: {preview_stats['sampled']}")


if __name__ == "__main__":
    main()
