from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import BERT_MODEL_DIR

def load_model(model_path=None):
    if model_path is None:
        model_path = BERT_MODEL_DIR
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()
    return tokenizer, model
