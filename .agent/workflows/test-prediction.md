---
description: Test the prediction API with sample reviews
---

# Test Prediction Workflow

This workflow helps you test the sentiment prediction functionality with various examples.

## Prerequisites

- BERT model trained (see `run-pipeline.md` workflow, Step 4)
- Virtual environment activated
- Model files exist in `models/bert/`

## Quick Test

### Single Prediction Test

```bash
python -c "from src.predict import predict_sentiment; sentiment, confidence = predict_sentiment('This product is amazing!'); print(f'Sentiment: {sentiment}, Confidence: {confidence:.2%}')"
```

**Expected Output:** `Sentiment: positive, Confidence: 95%` (approximate)

---

## Comprehensive Testing

### Test 1: Positive Review

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('This product exceeded all my expectations! Absolutely love it. Best purchase ever!'); print(f'✅ Positive Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `positive` with high confidence (>80%)

---

### Test 2: Negative Review

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('Terrible quality. Broke after one day. Complete waste of money. Very disappointed.'); print(f'✅ Negative Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `negative` with high confidence (>80%)

---

### Test 3: Neutral Review

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('The product is okay. Nothing special but it works as described.'); print(f'✅ Neutral Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `neutral` with moderate confidence (>60%)

---

### Test 4: Multilingual - French

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('Ce produit est excellent! Je le recommande vivement.'); print(f'✅ French Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `positive` (French: "This product is excellent! I highly recommend it.")

---

### Test 5: Multilingual - German

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('Sehr schlechte Qualität. Nicht empfehlenswert.'); print(f'✅ German Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `negative` (German: "Very poor quality. Not recommended.")

---

### Test 6: Multilingual - Spanish

```bash
python -c "from src.predict import predict_sentiment; result = predict_sentiment('Producto normal, nada especial.'); print(f'✅ Spanish Test: {result[0]} ({result[1]:.2%})')"
```

**Expected:** `neutral` (Spanish: "Normal product, nothing special.")

---

### Test 7: Batch Prediction

Test multiple reviews at once:

```bash
python -c "from src.predict import predict_sentiment; reviews = ['Great product!', 'Terrible experience', 'It is okay']; for review in reviews: sentiment, conf = predict_sentiment(review); print(f'{review[:20]:20} -> {sentiment:8} ({conf:.0%})')"
```

**Expected Output:**

```
Great product!       -> positive (95%)
Terrible experience  -> negative (92%)
It is okay           -> neutral  (78%)
```

---

## Interactive Testing Script

Create a test script for manual testing:

```bash
python -c "
from src.predict import predict_sentiment

print('=== ShopEase Sentiment Prediction Test ===')
print('Enter reviews to test (or \"quit\" to exit)')
print()

while True:
    review = input('Review: ')
    if review.lower() in ['quit', 'exit', 'q']:
        break
    if review.strip():
        sentiment, confidence = predict_sentiment(review)
        print(f'  → Sentiment: {sentiment} (Confidence: {confidence:.2%})')
        print()
"
```

**Usage:** Type reviews and see predictions in real-time

---

## Verification Checklist

Test that the model correctly identifies:

- [ ] Clearly positive reviews (>80% confidence)
- [ ] Clearly negative reviews (>80% confidence)
- [ ] Neutral/mixed reviews (>60% confidence)
- [ ] French language reviews
- [ ] German language reviews
- [ ] Spanish language reviews
- [ ] Batch predictions (multiple reviews)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Run from project root directory:

```bash
cd c:/Users/Muham/Documents/GitHub/ShopEase
```

### Issue: "Can't load tokenizer for 'models/bert/'"

**Solution:** Train the model first:

```bash
python -c "from src.train import train_bert, load_data; X, y = load_data('data/processed/clean_reviews.csv'); train_bert(X[:500], y[:500])"
```

### Issue: Predictions seem random or incorrect

**Solution:**

1. Check if model was trained on sufficient data
2. Verify model files exist in `models/bert/`
3. Retrain with more samples or epochs

### Issue: Low confidence scores (<50%)

**Solution:** This can happen with:

- Ambiguous reviews (genuinely neutral)
- Very short text (not enough context)
- Mixed sentiment (both positive and negative)
- Languages not well-represented in training data

---

## Expected Behavior

### Good Predictions (High Confidence)

- Clear positive language → `positive` (>85%)
- Clear negative language → `negative` (>85%)
- Explicit neutral statements → `neutral` (>70%)

### Moderate Predictions (Medium Confidence)

- Mixed sentiment → 60-80% confidence
- Short reviews → 60-80% confidence
- Sarcasm or irony → May be misclassified

### Low Confidence (<60%)

- Very ambiguous text
- Contradictory statements
- Unusual language or slang

---

## Sample Test Dataset

Create a CSV file `test_reviews.csv` for batch testing:

```csv
review,expected_sentiment
"Amazing product! Highly recommend!",positive
"Worst purchase ever. Total disappointment.",negative
"It's okay, nothing special.",neutral
"Excellent qualité! Très satisfait!",positive
"Schrecklich! Nie wieder!",negative
"Producto aceptable.",neutral
```

Test with:

```bash
python -c "
import pandas as pd
from src.predict import predict_sentiment

df = pd.read_csv('test_reviews.csv')
df['predicted'], df['confidence'] = zip(*df['review'].apply(predict_sentiment))

correct = (df['predicted'] == df['expected_sentiment']).sum()
total = len(df)
accuracy = correct / total

print(df[['review', 'expected_sentiment', 'predicted', 'confidence']])
print(f'\nAccuracy: {accuracy:.2%} ({correct}/{total})')
"
```

---

## Performance Benchmarks

Expected performance on test data:

- **Accuracy:** 85-95% on clear sentiment
- **Precision:** 80-90% per class
- **Recall:** 80-90% per class
- **F1-Score:** 80-90% per class

If your results are significantly lower:

1. Retrain with more data
2. Increase training epochs
3. Check data quality
4. Verify preprocessing is working correctly

---

## Next Steps

After successful testing:

1. Try the Streamlit dashboard: `streamlit run dashboards/streamlit_app.py`
2. Test with your own real customer reviews
3. Upload a CSV file for batch processing
4. Integrate the prediction API into your application

## Estimated Time

- **Quick test:** 1 minute
- **Comprehensive testing:** 5 minutes
- **Interactive testing:** As long as you want to experiment
- **Batch CSV testing:** 2-3 minutes
