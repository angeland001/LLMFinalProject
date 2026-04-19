# PEFT for VLM Medical Image Classification (Course Final Project)

This project provides a clean, reproducible codebase to compare four adaptation strategies for CLIP on chest X-ray classification:

1. **Zero-shot** prompt-based inference
2. **Linear probe** (frozen encoder + trainable classifier)
3. **LoRA fine-tuning** (PEFT on CLIP vision attention modules)
4. **Full fine-tuning** baseline

The initial target setup is **RSNA binary classification (normal vs pneumonia)** with **`openai/clip-vit-base-patch16`**.

---

## 1) Recommended project architecture

```text
LLMFinalProject/
├─ configs/
│  └─ base.yaml                   # Central experiment config
├─ data/
│  └─ rsna/                       # You place CSV splits + images here
├─ docs/
│  ├─ ai_prompts_used.txt         # AI-assisted prompt log for course submission
│  └─ implementation_steps.txt    # Chronological build log
├─ outputs/
│  ├─ experiment_summary.csv      # Aggregate results table
│  └─ plots/                      # Generated plots
├─ scripts/
│  ├─ run_experiment.py           # Run one method
│  ├─ run_all_experiments.py      # Run all methods in sequence
│  └─ plot_results.py             # Generate summary plots
├─ src/
│  ├─ data/
│  │  └─ rsna_dataset.py          # Dataset and dataloader creation
│  ├─ eval/
│  │  └─ zero_shot.py             # Zero-shot CLIP evaluation
│  ├─ models/
│  │  └─ clip_wrapper.py          # CLIP factory + LoRA utilities + classifier
│  ├─ trainers/
│  │  └─ train.py                 # Supervised train/eval loops
│  └─ utils/
│     ├─ io.py                    # YAML/JSON/CSV helpers
│     ├─ metrics.py               # Accuracy/AUROC/F1 + param counts
│     └─ reproducibility.py       # seed + device resolution
├─ requirements.txt
└─ README.md
```

---

## 2) What each file does (high-level)

- `configs/base.yaml`: single source of truth for dataset paths, model choices, hyperparameters, LoRA settings, and output directories.
- `src/data/rsna_dataset.py`: robust dataset loader with missing file/column checks.
- `src/models/clip_wrapper.py`: CLIP construction, freezing helper, LoRA adapter injection, and binary classifier head.
- `src/eval/zero_shot.py`: prompt-template based zero-shot evaluation.
- `src/trainers/train.py`: reusable training function for linear probe / LoRA / full fine-tune.
- `scripts/run_experiment.py`: one entrypoint for fair comparisons across methods.
- `scripts/run_all_experiments.py`: convenience orchestrator to run all variants.
- `scripts/plot_results.py`: builds comparison figures from aggregate CSV.
- `docs/ai_prompts_used.txt`: submission-ready AI usage log.
- `docs/implementation_steps.txt`: chronological implementation trail.

---

## 3) Suggested implementation order

1. Create config + reproducibility + I/O utilities.
2. Create dataset loader and verify split files exist.
3. Add CLIP backbone loader and simple classifier head.
4. Add zero-shot evaluator.
5. Add supervised training loop and metrics.
6. Add experiment entrypoint (`run_experiment.py`).
7. Add batch runner and plotting script.
8. Run RSNA baseline experiments and inspect summary table.
9. (Optional extension) Add BiomedCLIP and/or CheXpert split adapters.

---

## 4) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5) Prepare RSNA data

Expected CSV schema:
- `image_path` (relative to `data/rsna/images` or absolute)
- `label` (0 for normal, 1 for pneumonia)

Set paths in `configs/base.yaml`:
- `data.train_csv`
- `data.val_csv`
- `data.test_csv`
- `data.image_root`

---

## 6) Example commands

### Zero-shot
```bash
python scripts/run_experiment.py --config configs/base.yaml --method zero_shot
```

### Linear probe
```bash
python scripts/run_experiment.py --config configs/base.yaml --method linear_probe
```

### LoRA (single rank)
```bash
python scripts/run_experiment.py --config configs/base.yaml --method lora --lora-rank 8
```

### Full fine-tune
```bash
python scripts/run_experiment.py --config configs/base.yaml --method full_finetune
```

### Run all in sequence
```bash
python scripts/run_all_experiments.py
```

### Build plots + comparison visuals
```bash
python scripts/plot_results.py
```

---

## 7) Outputs you get

- Per-run metrics JSONs under `outputs/<method>/<timestamp>/metrics.json`
- Rolling aggregate table: `outputs/experiment_summary.csv`
- Plots under `outputs/plots/`

Tracked fields include:
- accuracy
- AUROC
- F1
- trainable parameter count + percent
- training time
- max GPU memory (if CUDA)

---

## 8) Reproducibility and guardrails

- Global seed setup in `src/utils/reproducibility.py`
- GPU/CPU auto fallback
- Missing file / missing column checks in dataset code
- AUROC guard for single-class split edge case
- Controlled config-driven experiments

---

## 9) Optional extensions (if time allows)

- Add `--backbone biomedclip` option and compare with CLIP.
- Add CheXpert binary subset adapters.
- Add weighted loss / sampler for stronger class imbalance handling.
- Add confidence intervals across multiple random seeds.

