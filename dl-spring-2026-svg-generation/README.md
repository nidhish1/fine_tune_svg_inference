# SVG Pipeline Finalized Plan

## Finalized implementation plan (competition-aligned)

### Target representation

- Train on structured semantic tokens (from `svg_tokenization` pipeline), not raw SVG text only.
- Keep canonicalization deterministic and fixed across train/inference.

### Generation pipeline (scene-builder style)

1. **Layout/components stage**: predict global composition, object slots, and draw order.
2. **Primitive/details stage**: predict geometry parameters and style/color fields.
3. **Serialization stage**: deterministically convert structured scene to final SVG text.

### Losses to use now

- `L_ce`: cross-entropy on structured token/field targets.
- `L_valid`: invalidity penalty (`penalty_based_signal.py`).
- `L_visual`: render-based loss (`1-SSIM` + edge loss), weighted highest.
- `L_compact`: output length-ratio penalty against target SVG.
- `L_struct_proxy`: structure surrogate for TED (tag/attribute sequence supervision).

### What not to do initially

- Avoid direct TED optimization first (non-differentiable and high complexity).
- Avoid RL-style full leaderboard optimization in the first iteration.

### Inference-time constraints

- Constrained decoding on allowed structure/fields.
- Hard limits during decode (`<path> <= 256`, max chars, valid structure).
- Rule-based repair fallback after decode failure.

### Training schedule

- Phase A: `L_ce + L_valid`.
- Phase B: add `L_visual + L_compact`.
- Phase C: add `L_struct_proxy` and joint stage-wise training.
- Phase D (optional): preference tuning with valid > invalid near-miss pairs.

### Metrics to track per epoch

- validity rate
- visual proxy score (SSIM/edge)
- structural proxy score
- compactness error
- latency p50/p95
