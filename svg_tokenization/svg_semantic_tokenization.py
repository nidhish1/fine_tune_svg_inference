#!/usr/bin/env python3
"""Semantic/field tokenization for SVG.

This module converts SVG XML into structured semantic tokens and can
reconstruct SVG from those tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import re
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

GEOMETRY_ATTRS = {
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "width",
    "height",
    "points",
    "viewBox",
}

STYLE_LIKE_ATTRS = {
    "fill",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "fill-opacity",
    "opacity",
    "font-size",
}

PATH_CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
PATH_TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

PARAM_SPEC = {
    "M": ["X", "Y"],
    "L": ["X", "Y"],
    "T": ["X", "Y"],
    "H": ["X"],
    "V": ["Y"],
    "C": ["X1", "Y1", "X2", "Y2", "X", "Y"],
    "S": ["X2", "Y2", "X", "Y"],
    "Q": ["X1", "Y1", "X", "Y"],
    "A": ["RX", "RY", "ROT", "LAF", "SF", "X", "Y"],
    "Z": [],
}


@dataclass
class SemanticToken:
    kind: str
    name: str
    value: str


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _token(kind: str, name: str, value: str) -> SemanticToken:
    return SemanticToken(kind=kind, name=name, value=value)


def _parse_style(style_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (style_text or "").split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key:
            out[key] = val
    return out


def _parse_path_d(d_text: str) -> List[SemanticToken]:
    tokens: List[SemanticToken] = []
    seq = PATH_TOKEN_RE.findall(d_text or "")
    idx = 0
    current_cmd: Optional[str] = None
    group_idx = 0

    while idx < len(seq):
        item = seq[idx]
        if PATH_CMD_RE.fullmatch(item):
            current_cmd = item
            tokens.append(_token("path_cmd", "CMD", item))
            idx += 1
            if item.upper() == "Z":
                group_idx = 0
            continue

        if current_cmd is None:
            # Ignore malformed numeric prefix without command.
            idx += 1
            continue

        spec = PARAM_SPEC[current_cmd.upper()]
        if not spec:
            idx += 1
            continue

        pos = group_idx % len(spec)
        field = spec[pos]
        tokens.append(_token("path_param", field, item))
        group_idx += 1
        idx += 1

    return tokens


def svg_to_semantic_tokens(svg_text: str) -> List[SemanticToken]:
    """Convert SVG XML into semantic field tokens."""
    root = ET.fromstring(svg_text)
    out: List[SemanticToken] = []

    def walk(node: ET.Element) -> None:
        tag = _local_name(node.tag).upper()
        out.append(_token("structure", f"OPEN_{tag}", ""))
        out.append(_token("tag", "TAG", tag))

        attrs = dict(sorted(node.attrib.items(), key=lambda kv: _local_name(kv[0])))
        for raw_key, val in attrs.items():
            key = _local_name(raw_key)
            value = val.strip()

            if tag == "PATH" and key == "d":
                out.append(_token("geometry", "PATH_D_START", ""))
                out.extend(_parse_path_d(value))
                out.append(_token("geometry", "PATH_D_END", ""))
            elif key == "style":
                style_map = dict(sorted(_parse_style(value).items(), key=lambda kv: kv[0]))
                for style_k, style_v in style_map.items():
                    out.append(_token("style", style_k.upper().replace("-", "_"), style_v))
            elif key in GEOMETRY_ATTRS:
                out.append(_token("geometry", key.upper(), value))
            elif key in STYLE_LIKE_ATTRS:
                out.append(_token("style", key.upper().replace("-", "_"), value))
            else:
                out.append(_token("attr", key.upper(), value))

        text = (node.text or "").strip()
        if text:
            out.append(_token("content", "TEXT", text))

        for child in list(node):
            walk(child)

        out.append(_token("structure", f"CLOSE_{tag}", ""))

    walk(root)
    return out


def semantic_tokens_to_svg(tokens: Iterable[SemanticToken]) -> str:
    """Rebuild SVG from semantic tokens.

    This reconstructs element/tag structure and common attrs, including path `d`.
    """
    stack: List[ET.Element] = []
    root: Optional[ET.Element] = None

    current_path_parts: List[str] = []
    in_path_d = False

    def current() -> ET.Element:
        if not stack:
            raise ValueError("Invalid token stream: no open element for attribute token.")
        return stack[-1]

    def geometry_token_name_to_attr(name: str) -> str:
        if name == "VIEWBOX":
            return "viewBox"
        return name.lower()

    for t in tokens:
        if t.kind == "structure" and t.name.startswith("OPEN_"):
            tag = t.name.replace("OPEN_", "").lower()
            elem = ET.Element(tag)
            if stack:
                stack[-1].append(elem)
            else:
                root = elem
            stack.append(elem)
            continue

        if t.kind == "structure" and t.name.startswith("CLOSE_"):
            if not stack:
                raise ValueError("Invalid token stream: CLOSE without OPEN.")
            closing = t.name.replace("CLOSE_", "").lower()
            if stack[-1].tag != closing:
                raise ValueError(f"Invalid nesting: closing {closing} but top is {stack[-1].tag}.")
            stack.pop()
            continue

        if t.kind == "tag":
            continue

        if t.kind == "geometry" and t.name == "PATH_D_START":
            in_path_d = True
            current_path_parts = []
            continue

        if t.kind == "geometry" and t.name == "PATH_D_END":
            in_path_d = False
            if current().tag == "path":
                current().set("d", " ".join(current_path_parts).strip())
            continue

        if in_path_d and t.kind == "path_cmd":
            current_path_parts.append(t.value)
            continue

        if in_path_d and t.kind == "path_param":
            current_path_parts.append(t.value)
            continue

        if t.kind == "geometry":
            current().set(geometry_token_name_to_attr(t.name), t.value)
            continue

        if t.kind == "style":
            attr = t.name.lower().replace("_", "-")
            current().set(attr, t.value)
            continue

        if t.kind == "attr":
            current().set(t.name.lower(), t.value)
            continue

        if t.kind == "content" and t.name == "TEXT":
            current().text = t.value
            continue

    if root is None:
        raise ValueError("Empty token stream.")
    if stack:
        raise ValueError("Invalid token stream: unclosed elements.")

    root.set("xmlns", SVG_NS)
    return ET.tostring(root, encoding="unicode", method="xml", short_empty_elements=True)


def tokens_to_lines(tokens: Iterable[SemanticToken]) -> List[str]:
    """Serialize tokens into stable text lines for model targets."""
    lines: List[str] = []
    for t in tokens:
        lines.append(f"{t.kind}|{t.name}|{t.value}")
    return lines


def lines_to_tokens(lines: Iterable[str]) -> List[SemanticToken]:
    """Parse line format back into semantic tokens."""
    out: List[SemanticToken] = []
    for line in lines:
        parts = line.rstrip("\n").split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid token line: {line!r}")
        out.append(SemanticToken(kind=parts[0], name=parts[1], value=parts[2]))
    return out

