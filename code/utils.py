"""
Shared utilities for the emotion classification project.

- EMOTION_LABELS: integer -> string label map used by the dair-ai emotion dataset.
- clean_text: light preprocessing pass shared between the classical and
  transformer pipelines.
- load_emotion_splits: download (once) and return the three dataset splits
  as pandas DataFrames.
"""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd

# The dair-ai/emotion dataset labels, in the integer order used by the dataset.
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
LABEL_TO_ID = {v: k for k, v in EMOTION_LABELS.items()}


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, collapse whitespace.

    We intentionally keep punctuation marks like ! and ? because they can
    carry emotional signal (e.g., "really?!" vs "really"). Heavy-handed
    cleaning tends to hurt classical models on this dataset.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def load_emotion_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the dair-ai/emotion dataset as (train, val, test) DataFrames.

    Each frame has columns: text (str), label (int 0-5), label_name (str),
    clean_text (str).
    """
    # Imported lazily so utils.py can be imported without the `datasets` package
    # being installed (e.g., when only the Streamlit demo is being loaded).
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", "split")

    def to_frame(split_name: str) -> pd.DataFrame:
        df = ds[split_name].to_pandas()
        df["label_name"] = df["label"].map(EMOTION_LABELS)
        df["clean_text"] = df["text"].apply(clean_text)
        return df

    return to_frame("train"), to_frame("validation"), to_frame("test")


if __name__ == "__main__":
    # Smoke test: print the shape and label distribution of each split.
    train, val, test = load_emotion_splits()
    for name, df in [("train", train), ("validation", val), ("test", test)]:
        print(f"\n{name}: {len(df)} rows")
        print(df["label_name"].value_counts())
