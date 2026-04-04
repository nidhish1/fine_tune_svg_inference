#!/usr/bin/env python3
import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


csv.field_size_limit(sys.maxsize)

ALLOWED_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "defs",
    "use",
    "symbol",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "stop",
    "text",
    "tspan",
    "title",
    "desc",
    "style",
    "pattern",
    "marker",
    "filter",
}
DRAWABLE_TAGS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "tspan", "use"}
MAX_CHARS = 16000
MAX_PATHS = 256
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def to_float(value: str):
    if value is None:
        return None
    m = NUM_RE.search(str(value))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_points(points_text: str):
    vals = [float(x) for x in NUM_RE.findall(points_text or "")]
    if len(vals) < 2:
        return None
    xs = vals[0::2]
    ys = vals[1::2]
    if not xs or not ys:
        return None
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}


def path_bbox(path_sequence):
    xs, ys = [], []
    for tok in path_sequence:
        name = tok.get("name", "")
        v = to_float(tok.get("value"))
        if v is None:
            continue
        if name.startswith("X"):
            xs.append(v)
        elif name.startswith("Y"):
            ys.append(v)
    if not xs or not ys:
        return None
    return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}


def coarse_bbox(tag: str, geometry_fields: dict, path_sequence):
    tag = tag.lower()
    if tag == "rect":
        x = to_float(geometry_fields.get("X", "0")) or 0.0
        y = to_float(geometry_fields.get("Y", "0")) or 0.0
        w = to_float(geometry_fields.get("WIDTH", "0")) or 0.0
        h = to_float(geometry_fields.get("HEIGHT", "0")) or 0.0
        return {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}
    if tag == "circle":
        cx, cy, r = to_float(geometry_fields.get("CX")), to_float(geometry_fields.get("CY")), to_float(geometry_fields.get("R"))
        if None not in (cx, cy, r):
            return {"x_min": cx - r, "y_min": cy - r, "x_max": cx + r, "y_max": cy + r}
    if tag == "ellipse":
        cx, cy = to_float(geometry_fields.get("CX")), to_float(geometry_fields.get("CY"))
        rx, ry = to_float(geometry_fields.get("RX")), to_float(geometry_fields.get("RY"))
        if None not in (cx, cy, rx, ry):
            return {"x_min": cx - rx, "y_min": cy - ry, "x_max": cx + rx, "y_max": cy + ry}
    if tag == "line":
        x1, y1 = to_float(geometry_fields.get("X1")), to_float(geometry_fields.get("Y1"))
        x2, y2 = to_float(geometry_fields.get("X2")), to_float(geometry_fields.get("Y2"))
        if None not in (x1, y1, x2, y2):
            return {"x_min": min(x1, x2), "y_min": min(y1, y2), "x_max": max(x1, x2), "y_max": max(y1, y2)}
    if tag in {"polyline", "polygon"}:
        return parse_points(geometry_fields.get("POINTS", ""))
    if tag == "path":
        return path_bbox(path_sequence)

    x, y = to_float(geometry_fields.get("X")), to_float(geometry_fields.get("Y"))
    w, h = to_float(geometry_fields.get("WIDTH")), to_float(geometry_fields.get("HEIGHT"))
    if None not in (x, y, w, h):
        return {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}
    return None


def area(b):
    if not b:
        return 0.0
    w = max(0.0, float(b["x_max"]) - float(b["x_min"]))
    h = max(0.0, float(b["y_max"]) - float(b["y_min"]))
    return w * h


def intersection(a, b):
    if not a or not b:
        return 0.0
    x0 = max(float(a["x_min"]), float(b["x_min"]))
    y0 = max(float(a["y_min"]), float(b["y_min"]))
    x1 = min(float(a["x_max"]), float(b["x_max"]))
    y1 = min(float(a["y_max"]), float(b["y_max"]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def classify_error_type(svg_text: str) -> str:
    if not svg_text:
        return "empty_svg"
    if len(svg_text) > MAX_CHARS:
        return "char_overflow"
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return "xml_error"
    tag = root.tag.split("}", 1)[1] if "}" in root.tag else root.tag
    if tag != "svg":
        return "root_error"
    path_count = 0
    for node in root.iter():
        t = node.tag.split("}", 1)[1] if "}" in node.tag else node.tag
        if t == "path":
            path_count += 1
        if t not in ALLOWED_TAGS:
            return "tag_violation"
    if path_count > MAX_PATHS:
        return "path_overflow"
    return "valid"


def deterministic_fold_id(sample_id: str, num_folds: int = 5, seed: int = 42) -> int:
    key = f"{seed}:{sample_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:12], 16) % num_folds


def parse_semantic_tokens(token_text: str):
    lines = token_text.splitlines() if token_text else []
    stack = []
    completed = []
    z_counter = 0
    canvas = {}

    for line in lines:
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        kind, name, value = parts

        if kind == "structure" and name.startswith("OPEN_"):
            tag = name.replace("OPEN_", "").lower()
            node = {
                "tag": tag,
                "geometry_fields": {},
                "style_fields": {},
                "attrs": {},
                "path_sequence": [],
                "in_path_d": False,
                "object_id": None,
                "z_index": None,
            }
            if tag in DRAWABLE_TAGS:
                node["object_id"] = f"obj_{z_counter:04d}"
                node["z_index"] = z_counter
                z_counter += 1
            stack.append(node)
            continue

        if kind == "structure" and name.startswith("CLOSE_"):
            if not stack:
                continue
            node = stack.pop()
            if node["tag"] in DRAWABLE_TAGS and node["object_id"] is not None:
                bbox = coarse_bbox(node["tag"], node["geometry_fields"], node["path_sequence"])
                node["coarse_bbox"] = {k: round(v, 4) for k, v in bbox.items()} if bbox else None
                completed.append(node)
            continue

        if not stack:
            continue
        cur = stack[-1]

        if kind == "geometry" and name == "PATH_D_START":
            cur["in_path_d"] = True
            continue
        if kind == "geometry" and name == "PATH_D_END":
            cur["in_path_d"] = False
            continue
        if cur["in_path_d"] and kind in {"path_cmd", "path_param"}:
            cur["path_sequence"].append({"kind": kind, "name": name, "value": value})
            continue

        if kind == "geometry":
            cur["geometry_fields"][name] = value
            if cur["tag"] == "svg":
                if name == "WIDTH":
                    canvas["width"] = value
                elif name == "HEIGHT":
                    canvas["height"] = value
                elif name == "VIEWBOX":
                    canvas["viewBox"] = value
            continue
        if kind == "style":
            cur["style_fields"][name] = value
            continue
        if kind == "attr":
            cur["attrs"][name] = value
            continue

    object_slots = []
    detail_objects = {}
    draw_order = []
    for obj in completed:
        oid = obj["object_id"]
        object_slots.append({"object_id": oid, "tag": obj["tag"], "z_index": obj["z_index"], "coarse_bbox": obj.get("coarse_bbox")})
        draw_order.append(oid)
        detail_objects[oid] = {
            "primitive_type": obj["tag"],
            "geometry_fields": {**obj["geometry_fields"], "path_sequence": obj["path_sequence"] if obj["path_sequence"] else []},
            "style_fields": obj["style_fields"],
        }

    layout_target = {"canvas": canvas, "object_slots": object_slots, "draw_order": draw_order}
    detail_target = {"objects": detail_objects}
    return layout_target, detail_target


def visibility_ratio(layout_target):
    slots = layout_target.get("object_slots", []) or []
    draw_order = layout_target.get("draw_order", []) or []
    slot_map = {s.get("object_id"): s for s in slots if s.get("object_id")}
    out = {}
    for i, oid in enumerate(draw_order):
        bbox = (slot_map.get(oid) or {}).get("coarse_bbox")
        a = area(bbox)
        if a <= 0:
            out[oid] = None
            continue
        occluded = 0.0
        for above in draw_order[i + 1 :]:
            occluded += intersection(bbox, (slot_map.get(above) or {}).get("coarse_bbox"))
        out[oid] = round(max(0.0, min(1.0, max(0.0, a - occluded) / a)), 4)
    return out


def overlap_graph(layout_target):
    slots = layout_target.get("object_slots", []) or []
    draw_order = layout_target.get("draw_order", []) or []
    idx = {oid: i for i, oid in enumerate(draw_order)}
    edges = []
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            a, b = slots[i], slots[j]
            aid, bid = a.get("object_id"), b.get("object_id")
            if not aid or not bid:
                continue
            ia, ib = area(a.get("coarse_bbox")), area(b.get("coarse_bbox"))
            inter = intersection(a.get("coarse_bbox"), b.get("coarse_bbox"))
            if inter <= 0:
                continue
            union = max(1e-9, ia + ib - inter)
            edges.append(
                {
                    "a": aid,
                    "b": bid,
                    "iou": round(inter / union, 4),
                    "overlap_a": round(inter / max(1e-9, ia), 4),
                    "overlap_b": round(inter / max(1e-9, ib), 4),
                    "z_relation": "a_below_b" if idx.get(aid, 0) < idx.get(bid, 0) else "a_above_b",
                }
            )
    return edges


def path_command_histogram(detail_target):
    objects = (detail_target or {}).get("objects", {}) or {}
    counts, total = {}, 0
    for obj in objects.values():
        seq = ((obj.get("geometry_fields", {}) or {}).get("path_sequence", []) or [])
        for token in seq:
            if token.get("kind") == "path_cmd" and token.get("name") == "CMD":
                cmd = str(token.get("value", "")).upper()
                if not cmd:
                    continue
                counts[cmd] = counts.get(cmd, 0) + 1
                total += 1
    norm = {k: round(v / total, 6) for k, v in counts.items()} if total > 0 else {}
    return {"counts": counts, "normalized": norm, "total_cmds": total}


def structure_proxy_sequence(svg_text):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return ["PARSE_ERROR"]

    seq = []

    def local(tag):
        return tag.split("}", 1)[1] if "}" in tag else tag

    def walk(node):
        tag = local(node.tag).lower()
        seq.append(f"OPEN:{tag}")
        for attr in sorted(node.attrib.keys()):
            seq.append(f"ATTR:{local(attr).lower()}")
        for child in list(node):
            walk(child)
        seq.append(f"CLOSE:{tag}")

    walk(root)
    return seq


def compactness_bucket(svg_text):
    n = len(svg_text or "")
    if n < 1000:
        b = "very_short"
    elif n < 3000:
        b = "short"
    elif n < 7000:
        b = "medium"
    elif n < 12000:
        b = "long"
    else:
        b = "very_long"
    return {"char_len": n, "bucket": b}


def build_final_training(dsl_input: Path, svg_input: Path, output_csv: Path, scene_version: str, num_folds: int, split_seed: int):
    with dsl_input.open("r", newline="", encoding="utf-8") as tok_f, svg_input.open("r", newline="", encoding="utf-8") as svg_f, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as out_f:
        tok_reader = csv.DictReader(tok_f)
        svg_reader = csv.DictReader(svg_f)
        fields = [
            "id",
            "prompt",
            "layout_target",
            "detail_target",
            "serialization_target",
            "visibility_ratio",
            "overlap_graph",
            "path_command_histogram",
            "structure_proxy_sequence",
            "compactness_target",
            "scene_version",
            "object_count",
            "validity_target",
            "fold_id",
        ]
        writer = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()
        rows = 0
        mismatches = 0

        for tok_row, svg_row in zip(tok_reader, svg_reader):
            rows += 1
            tid, sid = tok_row.get("id"), svg_row.get("id")
            if tid != sid:
                mismatches += 1
            row_id = sid or tid or ""
            prompt = svg_row.get("prompt", tok_row.get("prompt", ""))
            svg = svg_row.get("svg", "")

            layout_target, detail_target = parse_semantic_tokens(tok_row.get("svg_tokens", ""))
            vis = visibility_ratio(layout_target)
            graph = overlap_graph(layout_target)
            hist = path_command_histogram(detail_target)
            struct = structure_proxy_sequence(svg)
            compact = compactness_bucket(svg)
            object_count = len(layout_target.get("object_slots", []) or [])

            err = classify_error_type(svg)
            validity_target = {"error_type": err, "is_valid": err == "valid", "max_chars": MAX_CHARS, "max_paths": MAX_PATHS}

            writer.writerow(
                {
                    "id": row_id,
                    "prompt": prompt,
                    "layout_target": json.dumps(layout_target, separators=(",", ":")),
                    "detail_target": json.dumps(detail_target, separators=(",", ":")),
                    "serialization_target": svg,
                    "visibility_ratio": json.dumps(vis, separators=(",", ":")),
                    "overlap_graph": json.dumps(graph, separators=(",", ":")),
                    "path_command_histogram": json.dumps(hist, separators=(",", ":")),
                    "structure_proxy_sequence": json.dumps(struct, separators=(",", ":")),
                    "compactness_target": json.dumps(compact, separators=(",", ":")),
                    "scene_version": scene_version,
                    "object_count": object_count,
                    "validity_target": json.dumps(validity_target, separators=(",", ":")),
                    "fold_id": deterministic_fold_id(str(row_id), num_folds=num_folds, seed=split_seed),
                }
            )
    return rows, mismatches


def sample_csv(input_csv: Path, output_csv: Path, sample_size: int, seed: int):
    random.seed(seed)
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    sample = random.sample(rows, min(sample_size, len(rows)))
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample)
    return len(sample)


def preview_all_cols(input_csv: Path, output_html: Path, samples: int, seed: int):
    random.seed(seed)
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sampled = random.sample(rows, min(samples, len(rows)))

    def pretty_json(text):
        try:
            return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
        except Exception:
            return text

    cards = []
    for i, row in enumerate(sampled, 1):
        rid = html.escape(row.get("id", ""))
        prompt = html.escape(row.get("prompt", ""))
        svg = row.get("serialization_target", "")

        extra_cols = [
            "layout_target",
            "detail_target",
            "visibility_ratio",
            "overlap_graph",
            "path_command_histogram",
            "structure_proxy_sequence",
            "compactness_target",
            "validity_target",
        ]
        extras = "".join(
            f"<div class='panel'><h3>{col}</h3><pre>{html.escape(pretty_json(row.get(col, '')))}</pre></div>" for col in extra_cols
        )
        cards.append(
            f"""
<section class='card'>
  <div class='meta'><strong>Sample:</strong> {i} &nbsp; <strong>ID:</strong> {rid}</div>
  <div class='meta'><strong>Prompt:</strong> {prompt}</div>
  <div class='grid'>
    <div class='panel'><h3>serialization_target (render)</h3><div class='svg-box'>{svg}</div></div>
    <div class='panel'><h3>serialization_target (raw)</h3><pre>{html.escape(svg)}</pre></div>
  </div>
  <div class='grid'>
    <div class='panel'><h3>scene_version</h3><pre>{html.escape(str(row.get('scene_version', '')))}</pre></div>
    <div class='panel'><h3>object_count / fold_id</h3><pre>{html.escape(json.dumps({'object_count': row.get('object_count'), 'fold_id': row.get('fold_id')}, indent=2))}</pre></div>
  </div>
  <div class='grid'>{extras}</div>
</section>
"""
        )

    page = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Final Training Preview</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #f6f8fb; color: #111; }}
    .card {{ background: #fff; border: 1px solid #e4e8ef; border-radius: 12px; padding: 12px; margin-bottom: 12px; }}
    .meta {{ margin-bottom: 8px; font-size: 14px; line-height: 1.4; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 10px; }}
    .panel {{ border: 1px solid #e7ebf2; border-radius: 8px; padding: 8px; background: #fcfdff; }}
    .panel h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .svg-box {{ min-height: 220px; display: flex; align-items: center; justify-content: center; background: #fff; border: 1px dashed #cfd7e3; border-radius: 8px; padding: 8px; overflow: hidden; }}
    .svg-box svg {{ width: 200px; height: 200px; max-width: 100%; max-height: 100%; }}
    pre {{ margin: 0; max-height: 260px; overflow: auto; background: #0f172a; color: #e2e8f0; padding: 8px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; font-size: 12px; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>Final Training Preview</h1>
  <p>{len(sampled)} random samples from final training CSV with all columns.</p>
  {''.join(cards)}
</body>
</html>"""
    output_html.write_text(page, encoding="utf-8")
    return len(sampled)


def main():
    parser = argparse.ArgumentParser(description="Generate full final training features in one run.")
    parser.add_argument("--dsl-input", default="../svg_tokenization/tokenized_train_safe.csv")
    parser.add_argument("--svg-input", default="../svg_tokenization/final_saved_train_safe.csv")
    parser.add_argument("--output", default="final_training.csv")
    parser.add_argument("--scene-version", default="v1.2")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sample-output", default="final_training_1000.csv")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--preview-output", default="preview_final_training_100_all_cols.html")
    parser.add_argument("--preview-samples", type=int, default=100)
    parser.add_argument("--preview-seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    rows, mismatches = build_final_training(
        dsl_input=Path(args.dsl_input),
        svg_input=Path(args.svg_input),
        output_csv=out,
        scene_version=args.scene_version,
        num_folds=args.num_folds,
        split_seed=args.split_seed,
    )
    sample_n = sample_csv(out, Path(args.sample_output), sample_size=args.sample_size, seed=args.preview_seed)
    preview_n = preview_all_cols(Path(args.sample_output), Path(args.preview_output), samples=args.preview_samples, seed=args.preview_seed)

    print(f"Final training CSV written: {out}")
    print(f"Rows processed: {rows}, id_mismatches: {mismatches}")
    print(f"Sample CSV written: {args.sample_output} ({sample_n} rows)")
    print(f"Preview HTML written: {args.preview_output} ({preview_n} samples)")


if __name__ == "__main__":
    main()
