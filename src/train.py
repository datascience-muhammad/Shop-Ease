from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

from src.preprocessing import PROCESSED_DATA_PATH, preprocess
from src.config import (
    BASELINE_MODEL_DIR, BERT_MODEL_DIR, REPORTS_DIR, 
    TRAINING_REPORT_PATH, BERT_MODEL_NAME, 
    TRAIN_BATCH_SIZE, NUM_EPOCHS
)

MODEL_PATH = BASELINE_MODEL_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = BASELINE_MODEL_DIR / "tfidf_vectorizer.joblib"


def _ensure_data() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        preprocess()
    return pd.read_csv(PROCESSED_DATA_PATH)


def _split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X = df["final_text"].astype(str) if "final_text" in df.columns else df["clean_text"].astype(str)
    y = df["label"].astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def _train_model(X_train: pd.Series, y_train: pd.Series) -> Tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(X_vec, y_train)
    return vectorizer, model


def _evaluate(model: LogisticRegression, vectorizer: TfidfVectorizer, X_test: pd.Series, y_test: pd.Series) -> dict:
    X_vec = vectorizer.transform(X_test)
    preds = model.predict(X_vec)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "report": classification_report(y_testwh, preds, output_dict=True),
    }


def _persist(model: LogisticRegression, vectorizer: TfidfVectorizer) -> None:
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)


def _write_report(metrics: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    summary = [
        "# Training Report",
        "",
        f"**Accuracy:** {metrics['accuracy']:.4f}",
        f"**Macro F1:** {metrics['macro_f1']:.4f}",
        "",
        "## Classification Report",
        json.dumps(metrics["report"], indent=2),
    ]
    TRAINING_REPORT_PATH.write_text("\n".join(summary))


def train() -> dict:
    df = _ensure_data()
    X_train, X_test, y_train, y_test = _split(df)
    vectorizer, model = _train_model(X_train, y_train)
    metrics = _evaluate(model, vectorizer, X_test, y_test)
    _persist(model, vectorizer)
    _write_report(metrics)
    return metrics


def load_data(path: Path | str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    X = df["final_text"].astype(str).tolist() if "final_text" in df.columns else df["clean_text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    return X, y


def train_bert(X_train: list[str], y_train: list[int]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=3)
    encodings = tokenizer(X_train, padding=True, truncation=True)

    class TextDataset(torch.utils.data.Dataset):
        """
        A helper class to feed our tokenized data into the BERT model.
        Think of this as a 'loader' that translates Python lists into PyTorch tensors.
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            # Tell the trainer how many reviews we have
            return len(self.labels)

        def __getitem__(self, idx):
            # This method pulls ONE review at a time for the model to learn from.
            # We convert the tokenizer dictionary into PyTorch tensors.
            item = {}
            for key, val in self.encodings.items():
                item[key] = torch.tensor(val[idx])
            
            # Use the 'labels' key specifically for the sentiment class
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_dataset = TextDataset(encodings, y_train)
    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR / "checkpoints"),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=50,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))
    return {"status": "trained", "samples": len(X_train)}


if __name__ == "__main__":
    results = train()
    printable = {k: v for k, v in results.items() if k != "report"}
    print(json.dumps(printable, indent=2))
