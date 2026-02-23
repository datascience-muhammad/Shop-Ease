---
description: Run the complete ML pipeline end-to-end
---

# Run Complete Pipeline Workflow

This workflow executes the entire ShopEase sentiment analysis pipeline from data generation to model evaluation.

## Prerequisites

- Setup completed (see `setup.md` workflow)
- Virtual environment activated
- All dependencies installed

## Pipeline Steps

### Step 1: Generate Synthetic Data

Generate 1000 sample customer reviews for training:

```bash
python -c "from extras.generate_synthetic_data import generate_reviews; from src.utils import save_csv; df = generate_reviews(1000); save_csv(df, 'data/raw/raw_reviews.csv'); print(f'✅ Generated {len(df)} reviews')"
```

**Expected Output:** `✅ Generated 1000 reviews`

**What this does:** Creates synthetic multilingual customer reviews with sentiment labels (positive, neutral, negative)

---

### Step 2: Preprocess Data

Clean and normalize the text data:

```bash
python src/preprocessing.py
```

**Expected Output:**

- Progress messages about text cleaning
- Saved file: `data/processed/clean_reviews.csv`

**What this does:**

- Converts text to lowercase
- Removes special characters
- Detects language
- Lemmatizes words
- Removes stopwords
- Creates numeric labels (0=negative, 1=neutral, 2=positive)

---

### Step 3: Train Baseline Model (Optional)

Train a simple TF-IDF + Logistic Regression baseline:

```bash
python -c "from src.train import train; results = train(); print(f'✅ Baseline Accuracy: {results[\"accuracy\"]:.4f}')"
```

**Expected Output:** `✅ Baseline Accuracy: 0.75-0.85` (approximate)

**What this does:**

- Vectorizes text using TF-IDF
- Trains Logistic Regression classifier
- Saves model to `models/baseline/`
- Generates training report

---

### Step 4: Train BERT Model

Train the main DistilBERT multilingual model:

```bash
python -c "from src.train import train_bert, load_data; X, y = load_data('data/processed/clean_reviews.csv'); print('Training BERT model...'); result = train_bert(X[:500], y[:500]); print(f'✅ BERT trained on {result[\"samples\"]} samples')"
```

**Expected Output:**

- Training progress bars
- `✅ BERT trained on 500 samples`

**What this does:**

- Loads DistilBERT multilingual model
- Fine-tunes on your data (using 500 samples for speed)
- Saves model and tokenizer to `models/bert/`

**Note:** Using 500 samples for quick demo. For production, use full dataset.

**Time:** ~5-10 minutes on CPU, ~1-2 minutes on GPU

---

### Step 5: Evaluate Model

Evaluate the trained BERT model on test data:

```bash
python -c "from src.evaluate import evaluate; from src.model_loader import load_model; from src.train import load_data; from sklearn.model_selection import train_test_split; X, y = load_data('data/processed/clean_reviews.csv'); _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42); tokenizer, model = load_model(); print('Evaluating model...'); evaluate(model, X_test[:100], y_test[:100], tokenizer)"
```

**Expected Output:**

- Accuracy score (e.g., 0.85-0.95)
- Classification report (precision, recall, F1)
- Confusion matrix heatmap (displayed in popup window)

**What this does:**

- Loads trained model
- Makes predictions on test set
- Calculates performance metrics
- Visualizes confusion matrix

---

### Step 6: Test Single Prediction

Test the prediction API with a sample review:

```bash
python -c "from src.predict import predict_sentiment; sentiment, confidence = predict_sentiment('This product is absolutely amazing! Best purchase ever!'); print(f'✅ Sentiment: {sentiment} (confidence: {confidence:.2%})')"
```

**Expected Output:** `✅ Sentiment: positive (confidence: 95%)`

**What this does:**

- Loads trained model
- Preprocesses input text
- Returns sentiment prediction with confidence score

---

### Step 7: Launch Streamlit Dashboard

Start the interactive web dashboard:

```bash
streamlit run dashboards/streamlit_app.py
```

**Expected Output:**

- Local URL: `http://localhost:8501`
- Dashboard opens in browser

**What this does:**

- Launches interactive web interface
- Allows single review predictions
- Supports batch CSV upload
- Provides downloadable results

**To stop:** Press `Ctrl+C` in terminal

---

## Quick Run (All Steps)

To run the entire pipeline in one command:

```bash
python run_all.py
```

**Time:** ~10-15 minutes total

**What this does:** Executes steps 1-5 automatically

---

## Verification Checklist

After running the pipeline, verify:

- [ ] `data/raw/raw_reviews.csv` exists and contains 1000 rows
- [ ] `data/processed/clean_reviews.csv` exists with cleaned text
- [ ] `models/bert/` contains model files (config.json, pytorch_model.bin, tokenizer files)
- [ ] `reports/training_report.md` exists (if baseline was trained)
- [ ] Single prediction returns sensible results
- [ ] Streamlit dashboard launches successfully

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"

**Solution:** Activate virtual environment and reinstall requirements:

```bash
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### Issue: "Can't load tokenizer for 'models/bert/'"

**Solution:** Train the model first (Step 4)

### Issue: Out of memory during BERT training

**Solution:** Reduce batch size or number of samples:

```python
# In train.py, reduce batch_size from 16 to 8
# Or train on fewer samples: train_bert(X[:200], y[:200])
```

### Issue: Confusion matrix doesn't display

**Solution:** This is normal in some environments. The metrics are still printed to console.

### Issue: Streamlit shows "Model not found"

**Solution:** Ensure Step 4 (BERT training) completed successfully and `models/bert/` contains files

---

## Expected File Structure After Pipeline

```
ShopEase/
├── data/
│   ├── raw/
│   │   └── raw_reviews.csv          ✅ Generated
│   └── processed/
│       └── clean_reviews.csv        ✅ Generated
├── models/
│   ├── baseline/
│   │   ├── sentiment_model.joblib   ✅ Generated (if baseline trained)
│   │   └── tfidf_vectorizer.joblib  ✅ Generated (if baseline trained)
│   └── bert/
│       ├── config.json              ✅ Generated
│       ├── pytorch_model.bin        ✅ Generated
│       ├── tokenizer_config.json    ✅ Generated
│       └── vocab.txt                ✅ Generated
└── reports/
    ├── training_report.md           ✅ Generated (if baseline trained)
    └── evaluation_report.md         ✅ Generated (if evaluated)
```

---

## Next Steps

After running the pipeline:

1. Explore the Streamlit dashboard
2. Try predictions with your own reviews
3. Upload a CSV file for batch processing
4. Review the training and evaluation reports
5. Experiment with different model parameters

## Estimated Time

- **Full pipeline (run_all.py):** 10-15 minutes
- **Individual steps:** 1-3 minutes each
- **BERT training:** 5-10 minutes (depends on hardware)
