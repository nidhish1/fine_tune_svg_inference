# Training Plan

## 1) Training approach

Use a single model with multi-task supervision over structured targets (v1), instead of starting with multiple separate models.

- **Input**: `prompt`
- **Primary target**: structured sequence derived from:
  - `layout_target`
  - `detail_target`
- **Auxiliary targets**:
  - `validity_target`
  - `object_count`
  - `compactness_target`
  - `structure_proxy_sequence`

Use deterministic serialization for final SVG output/evaluation (`serialization_target`).

## 2) Composite loss (v1)

Use weighted multi-objective loss:

`L = w_ce*L_ce + w_valid*L_valid + w_obj*L_objcount + w_cmp*L_compact + w_str*L_struct`

### Loss components

- **`L_ce`**: token/field cross-entropy for structured output (main objective)
- **`L_valid`**: invalidity penalty (XML/root/tags/path limit/char limit)
- **`L_objcount`**: object count supervision (regression or bucket CE)
- **`L_compact`**: compactness/length supervision from `compactness_target`
- **`L_struct`**: structure proxy sequence supervision (TED surrogate)

### Add later (v2)

- **`L_visual`**: render-based loss (`1-SSIM` + edge proxy), computed periodically or on validation subset

## 3) Phase schedule

- **Phase A (stability)**: `L_ce + L_valid`
- **Phase B (quality)**: add `L_objcount + L_compact`
- **Phase C (structure)**: add `L_struct`
- **Phase D (optional)**: preference tuning with valid > invalid hard negatives

Do not enable all losses at once in the first run.

## 4) Initial loss weights

Starting values (tune by validation behavior):

- `w_ce = 1.0`
- `w_valid = 0.2`
- `w_obj = 0.1`
- `w_cmp = 0.1`
- `w_str = 0.15`

Adjustment guidance:

- too many invalid outputs -> increase `w_valid`
- bad length/compactness behavior -> increase `w_cmp`
- poor structural fidelity/order -> increase `w_str`

## 5) Training mechanics

- Use **Hugging Face Accelerate**
- Enable mixed precision (`bf16`/`fp16`) and gradient accumulation
- Use deterministic split via `fold_id`
- Keep constrained decoding for inference-time first (not mandatory in train loop v1)

## 6) Metrics per epoch

- validity pass rate
- structured CE loss
- object_count error (MAE or accuracy bucket)
- compactness error / bucket accuracy
- structure proxy accuracy
- render proxy on held-out subset (SSIM/edge)

## 7) Practical objective

Stabilize structural generation first, then progressively optimize quality and compactness while keeping validity high.

## 8) First approach (one-shot fast training)

If time is limited, run one fast training setup:

- `L_ce + L_valid + L_compact (+ L_objcount optional)`
- Recommended quick weights:
  - `w_ce = 1.0`
  - `w_valid = 0.15`
  - `w_cmp = 0.05`
  - `w_obj = 0.05` (optional)

Plain-language focus:

- `L_ce` = "say the right thing"
- `L_valid` = "say it legally"
- `L_compact` = "say it concisely"
- `L_objcount` = "draw roughly right amount"
