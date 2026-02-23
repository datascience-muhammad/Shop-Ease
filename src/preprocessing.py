"""Data cleaning utilities for the sentiment-analysis project."""

from pathlib import Path
import re

import pandas as pd
import spacy
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def _load_nlp() -> spacy.language.Language:
    for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
        try:
            return spacy.load(model)
        except OSError:
            continue
    nlp_fallback = spacy.blank("xx")
    return nlp_fallback


NLP = _load_nlp()


def _ensure_nltk() -> None:
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download("punkt")
    # Newer NLTK versions split tokenizer tables into 'punkt_tab'
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


def detect_language(text: str) -> str:
    """
    Identifies the language of the text. 
    Crucial for multilingual e-commerce reviews (English, French, German, etc.)
    """
    try:
        return detect(text)
    except Exception:
        return "unknown"


def clean_text(text: str) -> str:
    """
    Cleans raw text by removing noise.
    
    Processing Steps:
    1. Convert all characters to lowercase.
    2. Remove URLs and special characters.
    3. Remove extra whitespace.
    
    This helps the model focus on the actual words instead of punctuation.
    """
    text = str(text).lower()
    # The regex below keeps: letters (a-z), accented letters (À-ÿ), numbers, and spaces.
    # Everything else (emojis, punctuation, symbols) is removed.
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize(text: str) -> str:
    """
    Groups different forms of a word so they can be analyzed as a single item.
    Example: 'running', 'runs', 'ran' all become 'run'.
    
    We use spaCy (NLP library) to handle this linguistically.
    """
    doc = NLP(text)
    return " ".join(token.lemma_ if token.lemma_ else token.text for token in doc)


def remove_stopwords(text: str) -> str:
    """
    Removes common words (the, is, in) that don't carry much sentiment.
    Focuses the model on the meaningful keywords like 'great', 'bad', or 'broken'.
    """
    tokens = word_tokenize(text)
    sw = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in sw]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_nltk()
    if "review" not in df.columns:
        # Map common column name "text" to expected "review"
        if "text" in df.columns:
            df["review"] = df["text"]
        else:
            df["review"] = df.iloc[:, 0].astype(str)
    df["clean_text"] = df["review"].astype(str).apply(clean_text)
    df["language"] = df["clean_text"].apply(detect_language)
    df["lemma_text"] = df["clean_text"].apply(lemmatize)
    df["final_text"] = df["lemma_text"].apply(remove_stopwords)

    # Ensure numeric label column exists: 0=negative, 1=neutral, 2=positive
    if "label" not in df.columns:
        if "sentiment" in df.columns:
            mapping = {"negative": 0, "neutral": 1, "positive": 2}
            df["label"] = df["sentiment"].map(mapping).fillna(1).astype(int)
        elif "rating" in df.columns:
            df["label"] = df["rating"].apply(lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2)).astype(int)
        else:
            df["label"] = 1  # default neutral
    return df


def preprocess() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df = preprocess_dataframe(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


if __name__ == "__main__":
    preprocess()
