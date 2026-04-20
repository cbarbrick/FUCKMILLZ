"""
Train and evaluate the two classical models for the emotion-classification
project: Multinomial Naive Bayes on raw word counts, and Logistic Regression
on TF-IDF features.

Run:
    python train_classical.py

Writes:
    models/nb.joblib, models/logreg.joblib
    results/classical_metrics.json
    results/classical_report.txt   <- copy-paste ready for the paper
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB

from utils import EMOTION_LABELS, load_emotion_splits


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def fit_and_eval(name, vectorizer, classifier, train_df, test_df):
    """Fit a (vectorizer, classifier) pipeline and return metrics + the fitted objects."""
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    per_class = classification_report(
        y_test,
        preds,
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)],
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds).tolist()

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}")
    print(per_class)

    return {
        "name": name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": per_class,
        "confusion_matrix": cm,
    }, vectorizer, classifier


def main():
    print("Loading dair-ai/emotion splits ...")
    train_df, val_df, test_df = load_emotion_splits()
    print(
        f"Train: {len(train_df):,}  "
        f"Validation: {len(val_df):,}  "
        f"Test: {len(test_df):,}"
    )
    print("\nTraining class distribution:")
    print(train_df["label_name"].value_counts())

    results = {}

    # --- Naive Bayes on bag-of-words counts ---
    nb_metrics, nb_vec, nb_clf = fit_and_eval(
        "Multinomial Naive Bayes (BoW counts)",
        CountVectorizer(min_df=2),
        MultinomialNB(),
        train_df,
        test_df,
    )
    results["naive_bayes"] = nb_metrics
    dump((nb_vec, nb_clf), MODELS_DIR / "nb.joblib")

    # --- TF-IDF + Logistic Regression ---
    lr_metrics, lr_vec, lr_clf = fit_and_eval(
        "TF-IDF + Logistic Regression",
        TfidfVectorizer(
            min_df=2,
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            C=1.0,
            n_jobs=-1,
        ),
        train_df,
        test_df,
    )
    results["tfidf_logreg"] = lr_metrics
    dump((lr_vec, lr_clf), MODELS_DIR / "logreg.joblib")

    # --- Write machine-readable metrics ---
    with open(RESULTS_DIR / "classical_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Write a human-readable summary ready to paste into the paper ---
    lines = []
    lines.append("Classical emotion classification results on the dair-ai/emotion test set\n")
    lines.append(f"{'Model':<40}{'Accuracy':>12}{'Macro-F1':>12}")
    lines.append("-" * 64)
    lines.append(
        f"{'Multinomial Naive Bayes':<40}"
        f"{results['naive_bayes']['accuracy']:>12.4f}"
        f"{results['naive_bayes']['macro_f1']:>12.4f}"
    )
    lines.append(
        f"{'TF-IDF + Logistic Regression':<40}"
        f"{results['tfidf_logreg']['accuracy']:>12.4f}"
        f"{results['tfidf_logreg']['macro_f1']:>12.4f}"
    )
    lines.append("")
    lines.append("Per-class reports:")
    lines.append("")
    lines.append("Multinomial Naive Bayes")
    lines.append(results["naive_bayes"]["classification_report"])
    lines.append("")
    lines.append("TF-IDF + Logistic Regression")
    lines.append(results["tfidf_logreg"]["classification_report"])

    (RESULTS_DIR / "classical_report.txt").write_text("\n".join(lines))
    print("\nWrote:")
    print(f"  {RESULTS_DIR / 'classical_metrics.json'}")
    print(f"  {RESULTS_DIR / 'classical_report.txt'}")


if __name__ == "__main__":
    main()
