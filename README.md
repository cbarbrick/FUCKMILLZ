# CSC 4444G — Emotion Detection in Text

Final project for CSC 4444G, Spring 2026.
**Authors:** Charlotte Barbrick, Easton Kling, Oscar Gayle.

An end-to-end emotion classifier on the
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
dataset (six labels: sadness, joy, love, anger, fear, surprise). The
project compares three models: Multinomial Naive Bayes (bag-of-words),
TF-IDF + Logistic Regression, and a fine-tuned DistilBERT transformer.
A small Streamlit app demos the best-performing model on user-typed
text.

## Repository layout

```
.
├── paper/                 # ICML-style LaTeX report (4 pages + refs)
│   ├── main.tex           # the paper source
│   ├── refs.bib           # bibliography
│   ├── main.pdf           # compiled PDF (also in latest build)
│   └── icml2026.sty, ...  # ICML template files
├── code/                  # all project code
│   ├── README.md          # how to set up and run
│   ├── environment.yml    # Conda environment spec
│   ├── requirements.txt   # pip fallback
│   ├── utils.py           # shared preprocessing + dataset loader
│   ├── train_classical.py # Multinomial NB + TF-IDF + LR
│   ├── train_distilbert.py# DistilBERT fine-tuning
│   └── app.py             # Streamlit demo
└── README.md              # this file
```

## Running the code

See `code/README.md` for full environment setup and run instructions.
Short version:

```bash
cd code
conda env create -f environment.yml
conda activate emotion

python train_classical.py      # NB + TF-IDF+LR, <1 min on CPU
python train_distilbert.py     # DistilBERT, ~15 min on a GPU
streamlit run app.py           # open the demo at localhost:8501
```

Each training script writes a `results/*_report.txt` with the accuracy
and macro-F1 numbers that should be copied into Table 2 of `paper/main.tex`
(the cells currently show `__.__`).

## Building the paper

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Requires a TeX distribution (TeX Live or MiKTeX) with the `hyperref`,
`booktabs`, `microtype`, `cleveref`, and `tikz` packages.
