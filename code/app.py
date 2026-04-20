"""
Streamlit demo for the emotion classifier.

Loads the fine-tuned DistilBERT model from models/distilbert/ and lets the user
type a short message, then shows the predicted emotion and the per-class
probability as a horizontal bar chart.

Run:
    streamlit run app.py

If models/distilbert/ does not exist, the app falls back to the TF-IDF +
Logistic Regression model in models/logreg.joblib so the demo still works
without fine-tuning first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils import EMOTION_LABELS, clean_text

HERE = Path(__file__).resolve().parent
DISTILBERT_DIR = HERE / "models" / "distilbert"
LOGREG_PATH = HERE / "models" / "logreg.joblib"


@st.cache_resource(show_spinner=False)
def load_distilbert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
    model.eval()
    return tokenizer, model, torch


@st.cache_resource(show_spinner=False)
def load_logreg():
    from joblib import load

    vectorizer, classifier = load(LOGREG_PATH)
    return vectorizer, classifier


def predict_distilbert(text, tokenizer, model, torch):
    cleaned = clean_text(text)
    with torch.no_grad():
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs


def predict_logreg(text, vectorizer, classifier):
    cleaned = clean_text(text)
    x = vectorizer.transform([cleaned])
    probs = classifier.predict_proba(x)[0]
    return probs


def main():
    st.set_page_config(page_title="Emotion Detector", page_icon=":speech_balloon:")
    st.title("Emotion Detection in Text")
    st.write(
        "Type a short message below and the model will predict which of six "
        "emotions it expresses: sadness, joy, love, anger, fear, or surprise."
    )

    if DISTILBERT_DIR.exists():
        tokenizer, model, torch = load_distilbert()
        backend = "DistilBERT (fine-tuned)"
        predictor = lambda t: predict_distilbert(t, tokenizer, model, torch)
    elif LOGREG_PATH.exists():
        vectorizer, classifier = load_logreg()
        backend = "TF-IDF + Logistic Regression"
        predictor = lambda t: predict_logreg(t, vectorizer, classifier)
    else:
        st.error(
            "No trained model found. Run either `python train_classical.py` "
            "or `python train_distilbert.py` first."
        )
        return

    st.caption(f"Backend: {backend}")

    text = st.text_area(
        "Your message",
        value="I'm finally flying home tomorrow and I can't wait to see everyone!",
        height=110,
    )

    if st.button("Predict emotion", type="primary") and text.strip():
        probs = predictor(text)
        pred_idx = int(np.argmax(probs))
        pred_label = EMOTION_LABELS[pred_idx]

        st.subheader(f"Predicted emotion: {pred_label}")
        st.write(f"Confidence: {probs[pred_idx]:.1%}")

        df = pd.DataFrame(
            {
                "emotion": [EMOTION_LABELS[i] for i in range(len(probs))],
                "probability": probs,
            }
        ).sort_values("probability", ascending=True)

        st.bar_chart(df.set_index("emotion"), horizontal=True)


if __name__ == "__main__":
    main()
