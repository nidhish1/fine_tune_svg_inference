# Feature Engineering

## Core per-row fields (baseline)

Each training row should include:

- `id`
- `prompt`
- `layout_target`
  - global composition / canvas
  - object slots (fixed `N` or variable list)
  - draw order (`z` indices)
- `detail_target`
  - per-object primitive type
  - geometry params (bbox/path params/etc.)
  - style/color fields (`fill`, `stroke`, `opacity`, etc.)
- `serialization_target`
  - deterministic final SVG text (canonicalized)

This is the minimum staged schema for:

1. layout/components prediction
2. primitive/details prediction
3. deterministic serialization

## Why deterministic SVG target is still needed

Even with a DSL-first approach, keep canonicalized final SVG because:

- evaluation is SVG-based (visual/structural/compactness metrics)
- deterministic DSL -> SVG improves reproducibility
- canonical target reduces formatting noise in validation/debugging

## Recommended per-row additions (v2 schema)

To make training more robust, add:

- `scene_version`
  - schema/version control for backward compatibility
- `object_ids`
  - stable IDs linking `layout_target` <-> `detail_target`
- `object_count`
  - explicit supervision for slot/object count
- `validity_target`
  - expected validity flags (XML/root/tags/path limit/char limit)
- `compactness_target`
  - reference SVG length or normalized ratio anchor
- `structure_target`
  - lightweight structural sequence (tag/attr proxy for TED-like supervision)
- `train_weight` (optional)
  - sample-level weighting for noisy/ambiguous rows

## Dataset-level additions

These are not mandatory per-row, but strongly recommended:

- hard negative samples (invalid/near-miss outputs)
- `error_type` labels (`xml_error`, `tag_violation`, `path_overflow`, `char_overflow`, etc.)
- fixed split metadata (deterministic train/val fold ID)
- quality flags for suspicious or rendering-failure rows

## Practical note for 2B model

- keep prompt text short
- move rules/constraints to pipeline (validator + constrained decode + repair)
- use structured targets as the primary learning signal
- use canonical SVG as final output contract

