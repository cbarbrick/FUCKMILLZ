"""
Fine-tune DistilBERT on the dair-ai/emotion six-class classification task.

Run:
    python train_distilbert.py

Works on CPU for quick experiments, but a GPU (T4 on free Colab works fine)
is strongly recommended. With the defaults below, three epochs of training
take roughly 10-15 minutes on a T4.

Writes:
    models/distilbert/            <- the fine-tuned model + tokenizer
    results/distilbert_metrics.json
    results/distilbert_report.txt <- copy-paste ready for the paper
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import EMOTION_LABELS, clean_text


HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RESULTS_DIR = HERE / "results"
MODEL_OUT = MODELS_DIR / "distilbert"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading dair-ai/emotion ...")
    ds = load_dataset("dair-ai/emotion", "split")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        texts = [clean_text(t) for t in batch["text"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LEN,
        )

    tokenized = ds.map(preprocess, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTION_LABELS),
        id2label=EMOTION_LABELS,
        label2id={v: k for k, v in EMOTION_LABELS.items()},
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "distilbert_run"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning ...")
    trainer.train()

    print("Evaluating on the test split ...")
    test_output = trainer.predict(tokenized["test"])
    preds = np.argmax(test_output.predictions, axis=-1)
    labels = test_output.label_ids

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    per_class = classification_report(
        labels,
        preds,
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS)],
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds).tolist()

    print(f"\n=== DistilBERT fine-tuned ===")
    print(f"Accuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}")
    print(per_class)

    metrics = {
        "name": "DistilBERT (fine-tuned)",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": per_class,
        "confusion_matrix": cm,
        "training": {
            "model": MODEL_NAME,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_len": MAX_LEN,
        },
    }

    # Save model + tokenizer for the Streamlit demo
    trainer.save_model(str(MODEL_OUT))
    tokenizer.save_pretrained(str(MODEL_OUT))

    with open(RESULTS_DIR / "distilbert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "DistilBERT fine-tuning results on the dair-ai/emotion test set",
        "",
        f"Accuracy: {accuracy:.4f}",
        f"Macro-F1: {macro_f1:.4f}",
        "",
        "Per-class report:",
        per_class,
    ]
    (RESULTS_DIR / "distilbert_report.txt").write_text("\n".join(lines))

    print("\nWrote:")
    print(f"  {MODEL_OUT}")
    print(f"  {RESULTS_DIR / 'distilbert_metrics.json'}")
    print(f"  {RESULTS_DIR / 'distilbert_report.txt'}")


if __name__ == "__main__":
    main()
