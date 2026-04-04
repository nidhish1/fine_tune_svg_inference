#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SELF_CLOSING = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "stop"}
KNOWN_MARKERS = [
    "serialization_target:",
    "serialization:",
    "SVG target:",
    "layout_target:",
]


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

    # If there's an explicit svg start, start there.
    m_start = re.search(r"<svg\b", t, flags=re.IGNORECASE)
    if m_start:
        t = t[m_start.start() :]
    # If continuation starts after "<svg", restore it.
    elif t.startswith("version=") or t.startswith("xmlns") or t.startswith("height=") or t.startswith("width=") or t.startswith("viewBox="):
        t = "<svg " + t

    # Stop at first explicit closing svg if present.
    m_end = re.search(r"</svg>", t, flags=re.IGNORECASE)
    if m_end:
        t = t[: m_end.end()]

    return t.strip()


def strip_noise(text: str) -> str:
    t = text or ""
    # Remove repeated trailing ">" noise and garbage suffix.
    t = re.sub(r"(>\s*){5,}$", "", t)
    # Drop obvious JSON fragment tails that are not XML.
    t = t.split('"}', 1)[0]
    return t.strip()


def trim_to_last_complete_tag_boundary(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # If output is cut mid-tag, drop the dangling partial tag entirely.
    last_lt = t.rfind("<")
    last_gt = t.rfind(">")
    if last_lt > last_gt:
        t = t[:last_lt].rstrip()

    # If we still end with attribute/value junk (common truncation), keep only
    # content through the last full tag close.
    last_gt = t.rfind(">")
    if last_gt >= 0 and last_gt < (len(t) - 1):
        trailing = t[last_gt + 1 :].strip()
        if trailing:
            t = t[: last_gt + 1].rstrip()

    return t


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
        if fixed.endswith("/"):
            fixed += ">"
        else:
            fixed += " />"
    return (prefix + fixed).strip()


def auto_close_common_tags(text: str) -> str:
    t = text
    stack: list[str] = []

    for m in re.finditer(r"<(/?)([A-Za-z_:][\w:.-]*)([^>]*)>", t):
        closing, raw_name, rest = m.group(1), m.group(2), m.group(3)
        name = raw_name.split(":", 1)[-1].lower()

        if closing:
            # Pop until matching tag if possible.
            if name in stack:
                while stack:
                    top = stack.pop()
                    if top == name:
                        break
            continue

        # Opening tag.
        is_self_close = rest.strip().endswith("/") or name in SELF_CLOSING
        if not is_self_close:
            stack.append(name)

    # Close in reverse open order.
    for name in reversed(stack):
        t += f"</{name}>"

    # Ensure svg closed.
    if re.search(r"<svg\b", t, flags=re.IGNORECASE) and not re.search(r"</svg>", t, flags=re.IGNORECASE):
        t += "</svg>"

    return t


def try_parse(svg_text: str) -> tuple[bool, str]:
    try:
        ET.fromstring(svg_text)
        return True, "ok"
    except Exception as e:
        return False, f"xml_error:{type(e).__name__}"


def has_graphics_content(svg_text: str) -> bool:
    # Treat these as drawable/visible-content primitives for a strict utility metric.
    return bool(
        re.search(
            r"<(path|rect|circle|ellipse|line|polyline|polygon|text|image|use)\b",
            svg_text or "",
            flags=re.IGNORECASE,
        )
    )


def repair_one(generated_text: str) -> tuple[str, str, bool]:
    payload, origin = extract_serialization_payload(generated_text)
    candidate = crop_to_plausible_svg_region(payload)
    candidate = strip_noise(candidate)
    candidate = recover_dangling_shape_tag_tail(candidate)
    candidate = trim_to_last_complete_tag_boundary(candidate)

    if not candidate:
        return "", f"{origin}|empty_after_crop", False

    ok, reason = try_parse(candidate)
    if ok:
        return candidate, f"{origin}|direct_parse", True

    repaired = auto_close_common_tags(candidate)
    ok2, reason2 = try_parse(repaired)
    if ok2:
        return repaired, f"{origin}|autoclose", True

    # last attempt: trim to first close if now present, then parse again
    m_end = re.search(r"</svg>", repaired, flags=re.IGNORECASE)
    if m_end:
        repaired2 = repaired[: m_end.end()]
        ok3, reason3 = try_parse(repaired2)
        if ok3:
            return repaired2, f"{origin}|trimmed_close", True
        return "", f"{origin}|failed:{reason3}", False

    return "", f"{origin}|failed:{reason2}", False


def main():
    parser = argparse.ArgumentParser(description="Repair SVG outputs from model generations.")
    parser.add_argument("--input-csv", required=True, help="CSV with sample_id,prompt,generated_text")
    parser.add_argument("--output-csv", required=True, help="CSV output with repaired svg")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(in_path.open("r", newline="", encoding="utf-8")))
    out_rows = []
    repaired_count = 0
    graphics_count = 0
    unrecoverable = 0

    for r in rows:
        svg, status, ok = repair_one(r.get("generated_text", ""))
        if ok:
            repaired_count += 1
            if has_graphics_content(svg):
                graphics_count += 1
        else:
            unrecoverable += 1
        out_rows.append(
            {
                "sample_id": r.get("sample_id", ""),
                "prompt": r.get("prompt", ""),
                "svg": svg,
                "is_valid_xml": "1" if ok else "0",
                "has_graphics": "1" if (ok and has_graphics_content(svg)) else "0",
                "repair_status": status,
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sample_id", "prompt", "svg", "is_valid_xml", "has_graphics", "repair_status"],
        )
        w.writeheader()
        w.writerows(out_rows)

    total = len(out_rows)
    rate = (repaired_count / total) if total else 0.0
    graphics_rate = (graphics_count / total) if total else 0.0
    print(f"input: {in_path}")
    print(f"output: {out_path}")
    print(f"total: {total}")
    print(f"repaired_valid_xml: {repaired_count}")
    print(f"repaired_with_graphics: {graphics_count}")
    print(f"unrecoverable: {unrecoverable}")
    print(f"valid_rate: {rate:.4f}")
    print(f"graphics_rate: {graphics_rate:.4f}")


if __name__ == "__main__":
    main()
