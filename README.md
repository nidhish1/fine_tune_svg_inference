# SVG Training Notes

## Validator-aware training signals

Validator-aware training means making SVG validity part of training, not just a post-processing rule.

### 1) Penalty-based signal

- Generate outputs during training/validation.
- Run a validator (XML validity, allowed tags, max `<path>`, max chars).
- Add extra loss when output is invalid (larger violations get larger penalties).

### 2) Preference training (valid > invalid)

- For the same prompt, build output pairs:
  - one valid output
  - one invalid output
- Train the model to score/rank valid outputs above invalid ones (DPO/RLHF-style preference objective).

### What this teaches

- The model learns not only to match text targets, but also to produce structurally legal SVGs.

### Example penalty ideas

- Invalid XML: high penalty
- Disallowed tag present: high penalty
- `<path>` count over max (e.g. 300 > 256): proportional penalty
- Output too long (>16,000 chars): proportional penalty

### Practical training flow

1. Keep normal token loss (teacher forcing / next-token objective).
2. Add a validator-derived loss term (soft structural constraint).
3. Optionally run a preference-tuning phase with valid/invalid pairs.

In short: validator-aware training turns validity into a train-time supervision signal instead of relying only on an inference-time filter.

## Better SVG tokenization (semantic/field tokens)

Better SVG tokenization means representing SVG as structured graphics fields instead of only raw text bytes/subwords.

### What changes

- Instead of one plain text stream like:
  - `<path fill="#000" d="M 10 20 ..."/>`
- Use semantic pieces such as:
  - tag token: `PATH`
  - geometry fields: `CMD=M`, `X=10`, `Y=20`, ...
  - style fields: `FILL=#000`, `STROKE_WIDTH=2`, `OPACITY=1`
  - structure tokens: `OPEN_SVG`, `OPEN_G`, `CLOSE_G`, `CLOSE_SVG`

### Why this helps

- Reduces formatting noise from equivalent SVG string variants.
- Separates geometry/style meaning from XML syntax noise.
- Makes constrained decoding easier (you can constrain by field/tag type).
- Improves consistency for composition, layering, and style prediction.

### Practical note

- You can still render the final output back to valid SVG text.
- The key idea is changing the model's internal target representation during training.

## Lightweight repair head

A lightweight repair head is a second-stage fixer that runs after generation only when SVG output is invalid.

### Pipeline

1. Main model generates SVG.
2. Validator checks constraints (XML validity, root/tag allowlist, max `<path>`, max length).
3. If valid: return directly.
4. If invalid: run repair head (rules first, optional tiny model second).
5. Re-validate repaired SVG before returning.

### Repair options

- **Rule-based pass (recommended first):**
  - enforce `<svg>` root
  - remove disallowed tags
  - cap `<path>` count
  - trim to max length
  - normalize formatting safely
- **Tiny learned repair model (optional):**
  - input: invalid SVG
  - output: minimally edited valid SVG
  - use when rule-based pass is too destructive on hard cases

### Training data for learned repair (optional)

- Build pairs: `invalid_svg -> valid_svg`.
- Mix synthetic corruption + real invalid generations.
- Include corruption types such as malformed XML, disallowed tags, overflow limits, and namespace/root errors.

### Key design principle

- Keep repairs conservative: preserve geometry/style whenever possible and only modify violating parts.

### Monitoring

- invalid rate before repair
- repair success rate
- average edit distance after repair
- render similarity before/after repair
- repair latency overhead

## Structured target format (canonicalized SVG targets)

Train on canonicalized SVG targets instead of raw heterogeneous strings.

### Goal

- Reduce equivalent-string noise (different formatting, same rendering).
- Make train targets more consistent for better optimization.

### Canonical target properties

- normalized numeric formatting (safe float normalization)
- stable attribute ordering
- normalized style formatting/order
- valid XML with `<svg>` root
- allowed-tag enforcement and hard limits

### Training implication

- Keep the canonicalization pipeline fixed and deterministic so train/inference distributions stay aligned.

## Stage-wise generation (coarse to fine)

Generate SVG in stages instead of one-shot raw serialization.

### Recommended stages

1. **Layout/components stage:** plan global composition and major objects.
2. **Primitive/details stage:** add geometry details and style/color choices.
3. **Serialization stage:** emit final valid SVG text from structured representation.

### Prompting approach

- Expand one prompt into sub-prompts/tasks per stage (e.g., prompt -> prompt_layout + prompt_detail + prompt_serialize).
- Keep interfaces explicit between stages (component list, layer order, style slots).

## Constrained decoding (inference-time)

Apply grammar/schema constraints during token generation, not only post-hoc cleanup.

### During decode

- restrict tokens to valid XML/SVG grammar continuation
- enforce allowed tags only
- enforce hard limits in-stream (e.g., max `<path>`, max chars)

### Why this matters

- Prevents invalid outputs before they are fully generated.
- Reduces repair burden and improves reliability.

### Training-side note

- During training/validation, track invalid rows and optionally add invalidity penalties.
- Main structural hard enforcement remains an inference-time decoding policy.

## Rendered sequence awareness (draw order and occlusion)

Preserve and model draw order/layer signals so the model learns composition logic.

### Data/target handling

- preserve drawable element order in targets (no global sibling reordering)
- include optional layer-index markers/tokens for explicit z-order signal

### Evaluation hooks

- add render-based checks (pixel/SSIM style metrics) on validation samples
- monitor occlusion-sensitive failures where foreground/background ordering is wrong

## Main risks and gaps in current approach

- Scope risk: too many major changes at once (tokenization + multi-stage generation + constrained decode + repair + preference training).
- No phased success criteria: ideas exist, but go/no-go metrics per phase are not yet explicit.
- No cost/latency budget: constrained decode + repair can increase inference latency without a defined target.
- No failure taxonomy: invalid outputs are not yet grouped into actionable buckets (XML parse, tag violation, length overflow, draw-order error).
- Stage-wise generation is under-specified: interfaces between stages are not formalized (schema/fields/checkpoints).

## Executable rollout plan

### Strict phased rollout

1. Canonicalization + validator baseline
2. Constrained decoding
3. Rule-based repair pass
4. Validator-aware loss
5. Stage-wise generation
6. Semantic tokenization (last, highest complexity)

### Per-phase acceptance metrics

- invalid rate
- repair success rate
- render similarity
- latency p50 / p95

### Rollback criteria

- If quality drops beyond threshold or latency exceeds budget, disable the last added feature and revert to the previous stable phase.

## 2B-friendly defaults

- keep prompts short
- enforce constraints outside prompt text (decoder + validator + repair)
- prefer lightweight rules first
- avoid heavy multi-stage prompt chaining in a single long context

## Bottom line

The direction in this README is correct and modern. What is missing is operationalization: phased implementation order, explicit metrics, and latency-quality guardrails. Adding those turns this from a concept list into a practical roadmap.

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

