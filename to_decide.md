# To Decide: Training Finalization

Use this as a decision checklist before launching the training run.

---

## 1) Data split lock

- Final split strategy selected and frozen
- No train/val leakage confirmed

**Option A:** Use `fold_id == 0` for validation, others for training  
**Option B:** Use `fold_id in {0,1}` for validation, others for training  
**Option C:** K-fold training (multiple runs)

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  lets use 90 training and 10 percent for validations



---

## 2) Target format lock

- Structured output format finalized
- Tokenization format finalized
- Sequence length/truncation policy finalized

**Option A:** Single combined structured sequence target  
**Option B:** Multi-head targets per stage (layout/detail/serialize)  
**Option C:** Hybrid (combined target + auxiliary heads)

**Selected option:** [ ] A  [ ] B  [x] C  
**Notes / suggestions:** Use combined structured sequence as primary target, with auxiliary heads/losses for validity, object_count, compactness, and structure proxy. Best ROI under time constraints.  



---

## 3) Loss configuration lock

- One-shot loss set finalized
- Weights finalized
- Loss normalization/scaling defined

**Option A (fast):** `L_ce + L_valid + L_compact`  
**Option B (fast+):** `L_ce + L_valid + L_compact + L_objcount`  
**Option C (fuller):** add `L_struct` in same run

**Selected option:** [ ] A  [ ] B  [ ] C  
**Initial weights proposal:**  

- `w_ce = 1.0`
- `w_valid = 0.15`
- `w_cmp = 0.05`
- `w_obj = 0.05` (if Option B/C)

**Notes / suggestions:**  



---

## 4) Validation metrics and cadence

- Per-epoch metrics finalized
- Fixed visual sanity subset selected
- Best-checkpoint metric selected

**Option A:** Best by validity pass rate  
**Option B:** Best by weighted score (`validity + compactness + CE`)  
**Option C:** Best by render proxy on val subset

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  



---

## 5) Inference contract

- Decoding params fixed
- Validator + repair fallback fixed
- Output constraints enforced

**Option A:** Greedy decode + validator + repair  
**Option B:** Constrained decode + validator + repair  
**Option C:** Sampling decode + strict repair

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  



---

## 6) Runtime/training config

- Accelerate config finalized
- Precision mode selected (`bf16`/`fp16`)
- Batch size + grad accumulation fixed
- LR schedule and warmup fixed

**Option A:** Conservative stable config  
**Option B:** Aggressive faster config  
**Option C:** Two-stage LR schedule (warmup + decay)

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  



---

## 7) Baseline and ablation

- Baseline run (`L_ce` only) planned or completed
- One-shot objective run planned or completed
- Comparison criteria defined

**Option A:** Skip baseline (time-critical)  
**Option B:** Mini baseline only (short run)  
**Option C:** Full baseline + one-shot comparison

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  



---

## 8) Submission pipeline dry run

- Small end-to-end dry run completed
- Submission CSV format validated
- Output SVG validity spot-checked

**Option A:** Dry run on 50 samples  
**Option B:** Dry run on 200 samples  
**Option C:** Full validation split dry run

**Selected option:** [ ] A  [ ] B  [ ] C  
**Notes / suggestions:**  



---

## Final go/no-go

- Ready to train now
- Need one more config pass
- Need data/schema adjustment first

**Final decision:**  



