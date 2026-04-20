# Emotion Detection in Text — Code

Accompanying code for the CSC 4444G final project by
**Charlotte Barbrick, Easton Kling, and Oscar Gayle**.

This repository trains and evaluates three emotion classifiers on the
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
dataset (six labels: sadness, joy, love, anger, fear, surprise) and
ships a small Streamlit demo that classifies user-typed text.

## Contents

| File | Description |
|------|-------------|
| `utils.py` | Dataset loader and shared text-cleaning helper used by every script |
| `train_classical.py` | Trains Multinomial Naive Bayes (bag-of-words) and Logistic Regression (TF-IDF + bigrams) |
| `train_distilbert.py` | Fine-tunes `distilbert-base-uncased` on the six-class task |
| `app.py` | Streamlit demo. Uses DistilBERT if available, otherwise falls back to the LR model |
| `environment.yml` | Conda environment specification |
| `requirements.txt` | Fallback pip-only dependency list |

## 1. Environment setup (Miniconda / Anaconda)

Tested on macOS 14 (Apple Silicon), Ubuntu 22.04, and Windows 11 with
Miniconda. The classical models work on CPU; DistilBERT fine-tuning runs
on CPU but is much faster on a GPU (a free-tier Google Colab T4 is
sufficient).

```bash
# From this directory
conda env create -f environment.yml
conda activate emotion
```

If you prefer pip / venv:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### GPU install (optional)

`environment.yml` installs the CPU build of PyTorch. To use a CUDA GPU on
Linux, install the matching CUDA build of PyTorch *after* creating the
env, following the command on <https://pytorch.org/get-started/locally/>.
For example, for CUDA 12.1:

```bash
conda activate emotion
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

On Google Colab, skip `environment.yml` entirely and run:

```python
!pip install -q datasets transformers accelerate scikit-learn
```

## 2. Train the classical models

```bash
python train_classical.py
```

Downloads the dataset the first time (~3 MB), then trains Naive Bayes and
TF-IDF + Logistic Regression. Finishes in under a minute on a laptop CPU.

Outputs:

- `models/nb.joblib` and `models/logreg.joblib` — pickled vectorizer +
  classifier pairs used by the demo.
- `results/classical_metrics.json` — machine-readable metrics.
- `results/classical_report.txt` — copy-paste-ready table for the paper.

## 3. Fine-tune DistilBERT

```bash
python train_distilbert.py
```

Downloads `distilbert-base-uncased` (~270 MB) on first run, then
fine-tunes for three epochs. Runs on CPU but is slow; a free Colab T4 GPU
finishes in roughly 10–15 minutes.

Outputs:

- `models/distilbert/` — fine-tuned model + tokenizer used by the demo.
- `results/distilbert_metrics.json` / `results/distilbert_report.txt`.

## 4. Run the Streamlit demo

```bash
streamlit run app.py
```

Opens a local web app at <http://localhost:8501>. Type a short message
and the app shows the predicted emotion plus the per-class probability
bar chart. The demo automatically uses the fine-tuned DistilBERT model
if `models/distilbert/` exists, otherwise it falls back to the saved
Logistic Regression model so it works even if you skip step 3.

## 5. Updating the paper results table

After each training script finishes, open
`results/classical_report.txt` and `results/distilbert_report.txt` and
copy the accuracy and macro-F1 numbers into the Results table in
`paper/main.tex` (the row cells currently say `__.__`).

## Reproducibility notes

- Random seeds are fixed at 42 in `train_distilbert.py`; small
  variations between runs are still possible because of non-deterministic
  CUDA kernels on GPU.
- The classical models are deterministic given the fixed dataset splits.
- The HuggingFace dataset is versioned on the Hub; if the cached
  download is out of date, pass `download_mode="force_redownload"` to
  `load_dataset` in `utils.py`.
