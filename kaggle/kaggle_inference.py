#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

csv.field_size_limit(sys.maxsize)

SELF_CLOSING = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "stop"}
KNOWN_MARKERS = [
    "serialization_target:",
    "serialization:",
    "SVG target:",
    "layout_target:",
]


def build_prefix(prompt: str, output_mode: str) -> str:
    if output_mode == "serialization_prefill":
        return (
            f"Prompt:\n{prompt}\n\n"
            "Generate structured SVG targets.\n"
            "layout_target: {}\n"
            "detail_target: {}\n"
            "serialization_target: <svg xmlns=\"http://www.w3.org/2000/svg\" "
        )
    if output_mode == "svg_only":
        return (
            f"Prompt:\n{prompt}\n\n"
            "Generate a valid SVG image.\n"
            "Output only one complete <svg ...>...</svg> block.\n"
            "Do not output layout_target, detail_target, or any extra text.\n\n"
        )
    return f"Prompt:\n{prompt}\n\nGenerate structured SVG targets.\n\n"


def read_rows(csv_path: Path, id_col: str, prompt_col: str, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            sample_id = (row.get(id_col, "") or "").strip()
            prompt = (row.get(prompt_col, "") or "").strip()
            if not prompt:
                continue
            if not sample_id:
                sample_id = f"row_{i:06d}"
            rows.append({"sample_id": sample_id, "prompt": prompt})
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def extract_serialization_payload(text: str) -> tuple[str, str]:
    if not text:
        return "", "empty"
    best_idx = None
    best_marker = None
    for marker in KNOWN_MARKERS:
        i = text.find(marker)
        if i >= 0 and (best_idx is None or i < best_idx):
            best_idx = i
            best_marker = marker
    if best_idx is None or best_marker is None:
        return text, "no_known_marker"
    marker_name = best_marker.rstrip(":").replace(" ", "_")
    return text[best_idx + len(best_marker) :], f"from_{marker_name}"


def crop_to_plausible_svg_region(payload: str) -> str:
    t = (payload or "").strip()
    if not t:
        return ""
    m_start = re.search(r"<svg\b", t, flags=re.IGNORECASE)
    if m_start:
        t = t[m_start.start() :]
    elif t.startswith("version=") or t.startswith("xmlns") or t.startswith("height=") or t.startswith("width="):
        t = "<svg " + t

    m_end = re.search(r"</svg>", t, flags=re.IGNORECASE)
    if m_end:
        t = t[: m_end.end()]
    return t.strip()


def strip_noise(text: str) -> str:
    t = text or ""
    t = re.sub(r"(>\s*){5,}$", "", t)
    t = t.split('"}', 1)[0]
    return t.strip()


def recover_dangling_shape_tag_tail(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    last_lt = t.rfind("<")
    last_gt = t.rfind(">")
    if last_lt <= last_gt:
        return t

    tail = t[last_lt:].strip()
    prefix = t[:last_lt].rstrip()
    if not tail.startswith("<") or tail.startswith("</"):
        return prefix
    m = re.match(r"<([A-Za-z_:][\w:.-]*)\b", tail)
    if not m:
        return prefix
    name = m.group(1).split(":", 1)[-1].lower()
    if name not in {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "g"}:
        return prefix

    fixed = tail
    if fixed.count('"') % 2 == 1:
        fixed += '"'
    if fixed.count("'") % 2 == 1:
        fixed += "'"
    fixed = fixed.rstrip()
    if not fixed.endswith(">"):
        fixed += " />" if not fixed.endswith("/") else ">"
    return (prefix + fixed).strip()


def trim_to_last_complete_tag_boundary(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    last_lt = t.rfind("<")
    last_gt = t.rfind(">")
    if last_lt > last_gt:
        t = t[:last_lt].rstrip()
    last_gt = t.rfind(">")
    if last_gt >= 0 and last_gt < (len(t) - 1):
        trailing = t[last_gt + 1 :].strip()
        if trailing:
            t = t[: last_gt + 1].rstrip()
    return t


def auto_close_common_tags(text: str) -> str:
    t = text
    stack: list[str] = []
    for m in re.finditer(r"<(/?)([A-Za-z_:][\w:.-]*)([^>]*)>", t):
        closing, raw_name, rest = m.group(1), m.group(2), m.group(3)
        name = raw_name.split(":", 1)[-1].lower()
        if closing:
            if name in stack:
                while stack:
                    top = stack.pop()
                    if top == name:
                        break
            continue
        is_self_close = rest.strip().endswith("/") or name in SELF_CLOSING
        if not is_self_close:
            stack.append(name)
    for name in reversed(stack):
        t += f"</{name}>"
    if re.search(r"<svg\b", t, flags=re.IGNORECASE) and not re.search(r"</svg>", t, flags=re.IGNORECASE):
        t += "</svg>"
    return t


def try_parse(svg_text: str) -> bool:
    try:
        ET.fromstring(svg_text)
        return True
    except Exception:
        return False


def has_graphics_content(svg_text: str) -> bool:
    return bool(
        re.search(r"<(path|rect|circle|ellipse|line|polyline|polygon|text|image|use)\b", svg_text or "", re.IGNORECASE)
    )


def repair_generated_text(generated_text: str) -> tuple[str, str, bool, bool]:
    payload, origin = extract_serialization_payload(generated_text)
    candidate = crop_to_plausible_svg_region(payload)
    candidate = strip_noise(candidate)
    candidate = recover_dangling_shape_tag_tail(candidate)
    candidate = trim_to_last_complete_tag_boundary(candidate)
    if not candidate:
        return "", f"{origin}|empty_after_crop", False, False

    if try_parse(candidate):
        return candidate, f"{origin}|direct_parse", True, has_graphics_content(candidate)

    repaired = auto_close_common_tags(candidate)
    if try_parse(repaired):
        return repaired, f"{origin}|autoclose", True, has_graphics_content(repaired)

    return "", f"{origin}|failed", False, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-friendly SVG inference + repair pipeline")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model folder")
    parser.add_argument("--prompts-csv", required=True, help="CSV with test prompts")
    parser.add_argument("--id-col", default="sample_id", help="ID column name")
    parser.add_argument("--prompt-col", default="prompt", help="Prompt column name")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of prompts (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--output-mode", choices=["structured", "svg_only", "serialization_prefill"], default="structured")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--raw-output-csv", default="/kaggle/working/raw_generations.csv")
    parser.add_argument("--repaired-output-csv", default="/kaggle/working/repaired_generations.csv")
    parser.add_argument("--submission-csv", default="/kaggle/working/submission.csv")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    prompts_csv = Path(args.prompts_csv).resolve()
    raw_output = Path(args.raw_output_csv).resolve()
    repaired_output = Path(args.repaired_output_csv).resolve()
    submission_csv = Path(args.submission_csv).resolve()

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    repaired_output.parent.mkdir(parents=True, exist_ok=True)
    submission_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(prompts_csv, args.id_col, args.prompt_col, args.limit)
    if not rows:
        raise SystemExit("No valid prompt rows found.")

    print(f"rows_to_generate: {len(rows)}")
    print(f"loading_model_from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True}
    if args.dtype == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif args.dtype == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif args.dtype == "fp32":
        load_kwargs["torch_dtype"] = torch.float32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"device: {device}")

    raw_rows: list[dict[str, str]] = []
    repaired_rows: list[dict[str, str]] = []
    submission_rows: list[dict[str, str]] = []

    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            prompt = row["prompt"]
            prefix = build_prefix(prompt, args.output_mode)
            enc = tokenizer(prefix, return_tensors="pt").to(device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": args.do_sample,
            }
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            output_ids = model.generate(**enc, **gen_kwargs)
            continuation_ids = output_ids[0][enc["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)

            raw_rows.append(
                {"sample_id": row["sample_id"], "prompt": prompt, "generated_text": generated_text}
            )

            svg, status, is_valid_xml, has_graphics = repair_generated_text(generated_text)
            repaired_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt": prompt,
                    "svg": svg,
                    "is_valid_xml": "1" if is_valid_xml else "0",
                    "has_graphics": "1" if has_graphics else "0",
                    "repair_status": status,
                }
            )
            submission_rows.append({"id": row["sample_id"], "svg": svg})
            if i % 25 == 0 or i == len(rows):
                print(f"generated: {i}/{len(rows)}")

    with raw_output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "prompt", "generated_text"])
        w.writeheader()
        w.writerows(raw_rows)

    with repaired_output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_id", "prompt", "svg", "is_valid_xml", "has_graphics", "repair_status"],
        )
        w.writeheader()
        w.writerows(repaired_rows)

    with submission_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "svg"])
        w.writeheader()
        w.writerows(submission_rows)

    valid = sum(1 for r in repaired_rows if r["is_valid_xml"] == "1")
    graphics = sum(1 for r in repaired_rows if r["has_graphics"] == "1")
    total = len(repaired_rows)
    print(f"raw_output_csv: {raw_output}")
    print(f"repaired_output_csv: {repaired_output}")
    print(f"submission_csv: {submission_csv}")
    print(f"valid_xml: {valid}/{total} ({(valid/total if total else 0):.4f})")
    print(f"with_graphics: {graphics}/{total} ({(graphics/total if total else 0):.4f})")


if __name__ == "__main__":
    main()
