# ShopEase Teaching Path (9-Video Edition)

> **The Golden Path for Teaching This Project**

This document defines the **core teaching path** through the ShopEase codebase. The series is divided into 9 logical steps to provide a comprehensive learning experience.

---

## 🎯 Core Philosophy

**Teach the journey, not the entire codebase.**

Students should understand:

1. **Why** we need each step
2. **What** each component does
3. **How** to use it in practice

---

## 🎬 9-Video Series Plan

### Video 1: Project Overview & Setup (10 min)

- **Focus:** Business problem and technical stack.
- **Demo:** Show the final dashboard.
- **Setup:** Quick environment setup (`pip install`, spaCy download).

### Video 2: Data Understanding & EDA (12 min)

- **Focus:** Exploring raw data.
- **Notebook:** `02_Data_Understanding.ipynb`
- **Key Concepts:** Class imbalance, multilingual text, noise identification.

### Video 3: Data Cleaning & Preprocessing (15 min)

- **Focus:** Text normalization.
- **File:** `src/preprocessing.py`
- **Notebook:** `03_Data_Cleaning.ipynb`
- **Key Functions:** `clean_text`, `lemmatize`.

### Video 4: Baseline Modeling: TF-IDF & Logistic Regression (12 min)

- **Focus:** Traditional machine learning.
- **Notebook:** `04_Baseline_Modeling.ipynb`
- **Key Concepts:** Feature extraction, vectorization, performance benchmarks.

### Video 5: Deep Learning with BERT: Theory & Tokenization (15 min)

- **Focus:** Understanding transformers.
- **Notebook:** `05_BERT_Theory.ipynb`
- **Key Concepts:** Attention mechanism, WordPiece tokenization, Subwords.

### Video 6: Fine-Tuning Multilingual BERT (20 min)

- **Focus:** Training the advanced model.
- **Notebook:** `06_BERT_FineTuning.ipynb`
- **Key Concepts:** PyTorch datasets, TrainingArguments, Trainer API.

### Video 7: Model Evaluation & Metrics (12 min)

- **Focus:** Measuring success.
- **Notebook:** `07_Model_Evaluation.ipynb`
- **Key Concepts:** Precision, Recall, F1-Score, Confusion Matrix.

### Video 8: Building the Prediction API (10 min)

- **Focus:** Moving from training to inference.
- **Notebook:** `08_Prediction_API.ipynb`
- **Key Concepts:** Loading saved weights, Softmax for probabilities.

### Video 9: User Interface & Dashboard Walkthrough (15 min)

- **Focus:** Deployment and pipeline management.
- **Notebook:** `09_Interactive_Dashboard.ipynb`
- **File:** `dashboards/streamlit_app.py`
- **Key Concepts:** Streamlit UI, Single vs Batch processing.

---

## 📖 Teaching Principles

1. **Show, Don't Just Tell**: Always run code and show the resulting output.
2. **Build Incrementally**: Start with raw data and end with a production-ready dashboard.
3. **Simplify the Complex**: Use analogies for BERT and Transformers.
4. **Focus on Practicality**: Explain how this solves a real-world business problem.

---

## 🎯 Progress Check

- [x] All Notebooks created (00-01, 04-08).
- [x] Emojis removed for professional look.
- [x] Text simplified for student clarity.
- [x] PyTorch code simplified for easy explanation.
