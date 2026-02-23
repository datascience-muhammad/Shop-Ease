from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data Paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raw_reviews.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_reviews.csv"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "demo_reviews.csv"

# Model Paths
BASELINE_MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
BERT_MODEL_DIR = PROJECT_ROOT / "models" / "bert"

# Report Paths
REPORTS_DIR = PROJECT_ROOT / "reports"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.md"
EVALUATION_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"

# Model Config
BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_LABELS = 3

# Sentiment Mapping
LABEL_TO_NAME = {0: "Negative", 1: "Neutral", 2: "Positive"}
NAME_TO_LABEL = {v: k for k, v in LABEL_TO_NAME.items()}
