#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def extract_svg(text: str) -> str:
    m = re.search(r"<svg\b[\s\S]*?</svg>", text or "", flags=re.IGNORECASE)
    return m.group(0).strip() if m else ""


def is_valid_xml(svg_text: str) -> bool:
    if not svg_text:
        return False
    try:
        ET.fromstring(svg_text)
        return True
    except Exception:
        return False


def build_fix_prompt(prompt: str, generated_text: str, repaired_status: str) -> str:
    # Keep prompt concise but informative; bias toward emitting only valid SVG.
    return (
        "You are fixing an incomplete SVG generation.\n"
        "Output ONLY one complete valid SVG block and nothing else.\n"
        "Requirements:\n"
        "- Start with <svg ...> and end with </svg>\n"
        "- Use valid XML\n"
        "- Keep simple and faithful to prompt\n\n"
        f"Prompt:\n{prompt}\n\n"
        f"Previous generation (possibly broken):\n{generated_text[:3000]}\n\n"
        f"Repair status:\n{repaired_status}\n\n"
        "Final output:\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Retry unrecoverable rows with model-based SVG fixer.")
    parser.add_argument("--repaired-csv", required=True, help="CSV produced by repair_svg_outputs.py")
    parser.add_argument("--raw-csv", required=True, help="Original raw generations CSV")
    parser.add_argument("--model-path", required=True, help="Model path for finisher retry")
    parser.add_argument("--output-csv", required=True, help="Final merged CSV output")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Limit retries for quick smoke; 0 means all invalid")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repaired_path = Path(args.repaired_csv)
    raw_path = Path(args.raw_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repaired_rows = list(csv.DictReader(repaired_path.open("r", newline="", encoding="utf-8")))
    raw_rows = list(csv.DictReader(raw_path.open("r", newline="", encoding="utf-8")))
    raw_by_id = {r.get("sample_id", ""): r for r in raw_rows}

    invalid_rows = [r for r in repaired_rows if r.get("is_valid_xml") != "1"]
    if args.limit > 0:
        invalid_rows = invalid_rows[: args.limit]

    print(f"total_rows: {len(repaired_rows)}")
    print(f"invalid_rows_for_retry: {len(invalid_rows)}")

    if invalid_rows:
        print(f"loading tokenizer: {args.model_path}")
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        load_kwargs = {"trust_remote_code": True}
        if args.dtype == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif args.dtype == "fp16":
            load_kwargs["torch_dtype"] = torch.float16
        elif args.dtype == "fp32":
            load_kwargs["torch_dtype"] = torch.float32
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            load_kwargs["torch_dtype"] = torch.bfloat16

        print(f"loading model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"device: {device}")

        fixed_by_id: dict[str, dict] = {}
        with torch.no_grad():
            for i, row in enumerate(invalid_rows, 1):
                sid = row.get("sample_id", "")
                raw = raw_by_id.get(sid, {})
                prompt = raw.get("prompt", row.get("prompt", ""))
                gen = raw.get("generated_text", "")
                repair_status = row.get("repair_status", "")

                fix_prompt = build_fix_prompt(prompt, gen, repair_status)
                enc = tok(fix_prompt, return_tensors="pt").to(device)
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
                cont = out[0][enc["input_ids"].shape[1] :]
                txt = tok.decode(cont, skip_special_tokens=True)
                svg = extract_svg(txt)
                ok = is_valid_xml(svg)
                fixed_by_id[sid] = {
                    "finisher_text": txt,
                    "finisher_svg": svg,
                    "finisher_valid": "1" if ok else "0",
                }
                print(f"[{i}/{len(invalid_rows)}] sample_id={sid} finisher_valid={ok}")
    else:
        fixed_by_id = {}

    out_rows = []
    final_valid = 0
    for row in repaired_rows:
        sid = row.get("sample_id", "")
        base_svg = row.get("svg", "")
        base_ok = row.get("is_valid_xml") == "1"

        fin = fixed_by_id.get(sid)
        if base_ok:
            final_svg = base_svg
            final_ok = True
            source = "repair"
        elif fin and fin.get("finisher_valid") == "1":
            final_svg = fin.get("finisher_svg", "")
            final_ok = True
            source = "finisher"
        else:
            final_svg = ""
            final_ok = False
            source = "unrecoverable"

        if final_ok:
            final_valid += 1

        out_rows.append(
            {
                "sample_id": sid,
                "prompt": row.get("prompt", ""),
                "repair_status": row.get("repair_status", ""),
                "repair_valid_xml": row.get("is_valid_xml", "0"),
                "finisher_valid_xml": (fin.get("finisher_valid", "0") if fin else "0"),
                "final_valid_xml": "1" if final_ok else "0",
                "final_source": source,
                "svg": final_svg,
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "prompt",
                "repair_status",
                "repair_valid_xml",
                "finisher_valid_xml",
                "final_valid_xml",
                "final_source",
                "svg",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    total = len(out_rows)
    print(f"output: {out_path}")
    print(f"final_valid_xml: {final_valid}/{total} ({(final_valid/total if total else 0):.4f})")


if __name__ == "__main__":
    main()
