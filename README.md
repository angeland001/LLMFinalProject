# PEFT for Vision-Language Models on Chest X-ray Classification

This project is a **course-friendly, reproducible framework** for comparing four adaptation strategies on chest X-ray classification:

1. **Zero-shot CLIP inference** (prompt-based)
2. **Linear probing** (frozen encoder + trainable classifier)
3. **LoRA fine-tuning (PEFT)** on CLIP vision attention layers
4. **Full fine-tuning** baseline

The default setup targets **RSNA Pneumonia Detection (binary: pneumonia vs normal)** and OpenAI **CLIP ViT-B/16**. Optional extensions for BiomedCLIP and CheXpert are supported via config options.

---

## 1) Project goals

- Keep experiments modular and easy to explain in a final report
- Track:
  - Accuracy
  - AUROC
  - F1-score
  - Trainable parameter counts
  - Peak GPU memory usage
  - Wall-clock time
- Compare parameter efficiency vs performance (especially LoRA ranks 4, 8, 16)

---

## 2) Folder structure

```text
LLMFinalProject/
├── README.md
├── requirements.txt
├── ai_prompts_used.txt
├── implementation_steps.txt
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── zero_shot.yaml
│       ├── linear_probe.yaml
│       ├── lora_r4.yaml
│       ├── lora_r8.yaml
│       ├── lora_r16.yaml
│       └── full_finetune.yaml
├── scripts/
│   ├── run_experiment.py
│   ├── run_all.py
│   └── summarize_results.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── rsna_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_setup.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── zero_shot.py
│   │   ├── linear_probe.py
│   │   ├── lora_finetune.py
│   │   └── full_finetune.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── logging_utils.py
│       ├── params.py
│       └── reproducibility.py
└── outputs/
    └── .gitkeep
```

---

## 3) Setup

### Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Expected dataset format (RSNA)

Use a CSV with at least these columns:
- `image_path` (absolute or path relative to project root)
- `label` (0 for normal, 1 for pneumonia)
- `split` (`train`, `val`, `test`)

Example row:

```csv
image_path,label,split
/data/rsna/images/000001.dcm,1,train
```

> Note: DICOM and standard image formats are both supported.

---

## 4) Quick start

### 4.1 Zero-shot

```bash
python scripts/run_experiment.py --config configs/experiments/zero_shot.yaml
```

### 4.2 Linear probe

```bash
python scripts/run_experiment.py --config configs/experiments/linear_probe.yaml
```

### 4.3 LoRA (r=4 / 8 / 16)

```bash
python scripts/run_experiment.py --config configs/experiments/lora_r4.yaml
python scripts/run_experiment.py --config configs/experiments/lora_r8.yaml
python scripts/run_experiment.py --config configs/experiments/lora_r16.yaml
```

### 4.4 Full fine-tuning

```bash
python scripts/run_experiment.py --config configs/experiments/full_finetune.yaml
```

### 4.5 Run all experiments

```bash
python scripts/run_all.py
```

### 4.6 Build final comparison table and plots

```bash
python scripts/summarize_results.py --results_dir outputs/results
```

---

## 5) Reproducibility checklist

- Fixed random seeds across Python/NumPy/PyTorch
- Config-driven runs (YAML per experiment)
- Per-run output directory with:
  - `config_resolved.yaml`
  - `metrics.json`
  - `summary.txt`
  - optional prediction CSVs

---

## 6) Notes for course submission

This repo includes:
- `ai_prompts_used.txt` (AI-assisted development prompts in guided tone)
- `implementation_steps.txt` (chronological implementation log)

You can directly cite these in your final report appendix.

