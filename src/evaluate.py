import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, X_test: list[str], y_test: list[int], tokenizer):
    """
    Evaluates the model and displays a confusion matrix.
    """
    model.eval()
    encoded = tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**encoded)
        
    logits = outputs.logits.detach().cpu().numpy()
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
        
    predicted_classes = logits.argmax(axis=1)
    
    print("\n" + "="*30)
    print(f"Accuracy: {accuracy_score(y_test, predicted_classes):.4f}")
    print("-" * 30)
    print(classification_report(y_test, predicted_classes))
    print("="*30)
    
    # Visualization
    cm = confusion_matrix(y_test, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Sentiment Analysis Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
