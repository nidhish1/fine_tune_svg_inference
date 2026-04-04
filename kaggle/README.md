# Kaggle Inference

Use `kaggle_inference.py` inside a Kaggle notebook to run generation and write:

- raw generations CSV
- repaired generations CSV
- `submission.csv` (`id,svg`)

## 1) Install dependencies in notebook

```bash
pip install -q transformers accelerate sentencepiece
```

## 2) Run inference

```bash
python /kaggle/working/kaggle/kaggle_inference.py \
  --model-path /kaggle/input/YOUR_MODEL_DATASET/best_model \
  --prompts-csv /kaggle/input/YOUR_TEST_DATASET/test_processed_top500.csv \
  --id-col sample_id \
  --prompt-col prompt \
  --max-new-tokens 1536 \
  --dtype bf16 \
  --output-mode structured \
  --raw-output-csv /kaggle/working/raw_generations_top500.csv \
  --repaired-output-csv /kaggle/working/repaired_generations_top500.csv \
  --submission-csv /kaggle/working/submission_top500.csv
```

## Optional: try sampling for better shape quality

```bash
python /kaggle/working/kaggle/kaggle_inference.py \
  --model-path /kaggle/input/YOUR_MODEL_DATASET/best_model \
  --prompts-csv /kaggle/input/YOUR_TEST_DATASET/test_processed_top500.csv \
  --id-col sample_id \
  --prompt-col prompt \
  --max-new-tokens 1536 \
  --dtype bf16 \
  --output-mode structured \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --submission-csv /kaggle/working/submission_top500_sampled.csv
```

`submission.csv` format is:

- `id`
- `svg`
