import numpy as np
import torch
from src.model_loader import load_model
from src.config import LABEL_TO_NAME

# Load once for speed
try:
    tokenizer, model = load_model()
except Exception:
    tokenizer, model = None, None


def predict_sentiment(text: str | list[str]) -> tuple[str, float]:
    """
    Predict sentiment for given text.
    Returns: Tuple of (sentiment_label, confidence_score)
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
        
    if isinstance(text, str):
        inputs = [text]
    else:
        inputs = text
        
    encoded = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded)
        
    logits = outputs.logits.detach().cpu().numpy()
    logits = logits if logits.ndim == 2 else logits.reshape(1, -1)
    
    # Softmax for probabilities
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    
    label_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])
    
    return LABEL_TO_NAME[label_idx], confidence
