#!/usr/bin/env python3
"""Penalty-based validity signals for SVG generation training.

This module implements step (1):
1) Generate model outputs during train/validation.
2) Validate outputs against SVG constraints.
3) Add a severity-weighted penalty loss for invalid outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional
import xml.etree.ElementTree as ET


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


@dataclass
class ValidationConfig:
    max_chars: int = 16_000
    max_paths: int = 256
    allowed_tags: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.allowed_tags is None:
            self.allowed_tags = set(ALLOWED_TAGS)


@dataclass
class ValidationResult:
    is_valid: bool
    xml_valid: bool
    svg_root_valid: bool
    disallowed_tag_count: int
    path_count: int
    path_overflow: int
    char_count: int
    char_overflow: int


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def validate_svg(svg_text: str, cfg: ValidationConfig) -> ValidationResult:
    """Validate one SVG string against the configured hard constraints."""
    char_count = len(svg_text or "")
    char_overflow = max(0, char_count - cfg.max_chars)
    if not svg_text:
        return ValidationResult(
            is_valid=False,
            xml_valid=False,
            svg_root_valid=False,
            disallowed_tag_count=0,
            path_count=0,
            path_overflow=0,
            char_count=char_count,
            char_overflow=char_overflow,
        )

    try:
        root = ET.fromstring(svg_text)
        xml_valid = True
    except ET.ParseError:
        return ValidationResult(
            is_valid=False,
            xml_valid=False,
            svg_root_valid=False,
            disallowed_tag_count=0,
            path_count=0,
            path_overflow=0,
            char_count=char_count,
            char_overflow=char_overflow,
        )

    svg_root_valid = _local_name(root.tag) == "svg"
    disallowed_tag_count = 0
    path_count = 0

    for node in root.iter():
        tag = _local_name(node.tag)
        if tag not in cfg.allowed_tags:
            disallowed_tag_count += 1
        if tag == "path":
            path_count += 1

    path_overflow = max(0, path_count - cfg.max_paths)
    is_valid = (
        xml_valid
        and svg_root_valid
        and disallowed_tag_count == 0
        and path_overflow == 0
        and char_overflow == 0
    )
    return ValidationResult(
        is_valid=is_valid,
        xml_valid=xml_valid,
        svg_root_valid=svg_root_valid,
        disallowed_tag_count=disallowed_tag_count,
        path_count=path_count,
        path_overflow=path_overflow,
        char_count=char_count,
        char_overflow=char_overflow,
    )


def compute_invalidity_penalty(
    result: ValidationResult,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Convert validator outputs to a scalar penalty.

    Larger violations produce larger penalties.
    """
    w = {
        "invalid_xml": 3.0,
        "invalid_root": 1.5,
        "disallowed_tag": 0.5,
        "path_overflow": 0.05,
        "char_overflow": 0.001,
    }
    if weights:
        w.update(weights)

    penalty = 0.0
    if not result.xml_valid:
        penalty += w["invalid_xml"]
        # Parse failure means we cannot inspect structure reliably.
        return penalty
    if not result.svg_root_valid:
        penalty += w["invalid_root"]
    penalty += result.disallowed_tag_count * w["disallowed_tag"]
    penalty += result.path_overflow * w["path_overflow"]
    penalty += result.char_overflow * w["char_overflow"]
    return penalty


def batch_penalty_from_svgs(
    svgs: Iterable[str],
    cfg: Optional[ValidationConfig] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Mean penalty over a batch of SVG strings."""
    cfg = cfg or ValidationConfig()
    penalties: List[float] = []
    for svg in svgs:
        result = validate_svg(svg, cfg)
        penalties.append(compute_invalidity_penalty(result, weights))
    return sum(penalties) / max(1, len(penalties))


def penalty_augmented_loss(
    base_loss: float,
    generated_svgs: Iterable[str],
    lambda_penalty: float = 0.2,
    cfg: Optional[ValidationConfig] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Combine standard task loss with validator-based penalty.

    total_loss = base_loss + lambda_penalty * mean_invalidity_penalty
    """
    mean_penalty = batch_penalty_from_svgs(generated_svgs, cfg=cfg, weights=weights)
    total = base_loss + lambda_penalty * mean_penalty
    return {
        "base_loss": float(base_loss),
        "invalidity_penalty": float(mean_penalty),
        "lambda_penalty": float(lambda_penalty),
        "total_loss": float(total),
    }


def generate_outputs_for_validation(
    *,
    model_forward: Callable[[dict], object],
    decode_outputs: Callable[[object], List[str]],
    batch: dict,
) -> List[str]:
    """Hook for train/validation loop.

    - `model_forward`: typically model(batch) or model.generate(...)
    - `decode_outputs`: converts model outputs to SVG strings
    """
    raw_outputs = model_forward(batch)
    return decode_outputs(raw_outputs)

