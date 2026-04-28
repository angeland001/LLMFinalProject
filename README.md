
## Project goals

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
## Setup

### Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Expected dataset format

Use a CSV with at least these columns:
- `image_path` (absolute or path relative to project root)
- `label` (0 for normal, 1 for pneumonia)
- `split` (`train`, `val`, `test`)

Example row:

```csv
image_path,label,split
/data/rsna/images/000001.dcm,1,train
```


### Google Colab setup (no Kaggle required)


```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone <YOUR_REPO_URL> LLMFinalProject
%cd /content/LLMFinalProject
!bash scripts/colab_quickstart.sh
```

Then run any experiment (example: zero-shot):

```bash
!python scripts/run_experiment.py \
  --config configs/experiments/zero_shot.yaml \
  --num-workers 2 \
  --output-root /content/drive/MyDrive/LLMFinalProject/results
```

You can also override the dataset CSV path in Colab without editing YAML files:

```bash
!python scripts/run_experiment.py \
  --config configs/experiments/lora_r8.yaml \
  --data-csv-path /content/LLMFinalProject/data/rsna/metadata.csv \
  --num-workers 2 \
  --output-root /content/drive/MyDrive/LLMFinalProject/results
```

> This project now uses Hugging Face dataset download via `scripts/download_dataset.py`; Kaggle is not required.

---

## ) Quick start

### Zero-shot

```bash
python scripts/run_experiment.py --config configs/experiments/zero_shot.yaml
```

### Linear probe

```bash
python scripts/run_experiment.py --config configs/experiments/linear_probe.yaml
```

### LoRA (r=4 / 8 / 16)

```bash
python scripts/run_experiment.py --config configs/experiments/lora_r4.yaml
python scripts/run_experiment.py --config configs/experiments/lora_r8.yaml
python scripts/run_experiment.py --config configs/experiments/lora_r16.yaml
```

### Full fine-tuning

```bash
python scripts/run_experiment.py --config configs/experiments/full_finetune.yaml
```

### Run all experiments

```bash
python scripts/run_all.py
```

### Build final comparison table and plots

```bash
python scripts/summarize_results.py --results_dir outputs/results
```

---

## ) Reproducibility checklist

- Fixed random seeds across Python/NumPy/PyTorch
- Config-driven runs (YAML per experiment)
- Per-run output directory with:
  - `config_resolved.yaml`
  - `metrics.json`
  - `summary.txt`
  - optional prediction CSVs



