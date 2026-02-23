# ShopEase: Customer Review Sentiment Analysis

This project demonstrates an end-to-end Machine Learning pipeline for classifying customer feedback in an e-commerce context. It covers the full lifecycle from data generation and preprocessing to model training and deployment.

The system is built to handle multilingual reviews (English, French, German, Spanish) using a fine-tuned DistilBERT model.

## Project Architecture

The solution is divided into modular components:

- **Data Pipeline**: Synthetic data generation and automated cleaning/normalization using regex and spaCy.
- **Modeling**: Comparison between a Baseline model (TF-IDF + Logistic Regression) and a Deep Learning model (DistilBERT).
- **Evaluation**: Comprehensive performance metrics including Precision, Recall, F1-Score, and Confusion Matrix analysis.
- **Inference**: A REST-ready prediction module optimized for both single-instance and batch processing.
- **User Interface**: An interactive Streamlit dashboard for real-time testing and visualization.

## Directory Structure

```text
ShopEase/
├── data/               # Repository for raw and processed datasets
├── notebooks/          # Interactive Jupyter notebooks for education (Steps 02-09)
├── src/                # Source code for core logic (preprocessing, training, inference)
├── dashboards/         # Streamlit application source code
├── extras/             # Utility scripts for data generation
└── models/             # Serialized model artifacts and tokenizer configuration
```

## Curriculum & Modules

This project is structured as a 9-step educational series:

1.  **Overview & Setup**
2.  **Data Understanding**: Exploratory Data Analysis (EDA) and noise identification.
3.  **Data Cleaning**: Text preprocessing and label normalization.
4.  **Baseline Modeling**: Implementation of TF-IDF and Logistic Regression.
5.  **BERT Theory**: Conceptual overview of Transformers and Tokenization.
6.  **Fine-Tuning**: Training the DistilBERT model on custom data.
7.  **Evaluation**: Analyzing model performance metrics.
8.  **Prediction API**: Building the inference engine.
9.  **Interactive Dashboard**: Deployment using Streamlit.

## Getting Started

### Prerequisites

- Python 3.10+
- pip package manager

### Installation

1.  Clone the repository and navigate to the project root.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the spaCy language model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

### Running the Educational Notebooks

Start the Jupyter environment to follow the modules sequentially:

```bash
jupyter notebook notebooks/
```

### Analyzing Data via Dashboard

Launch the web interface to interact with the trained model:

```bash
streamlit run dashboards/streamlit_app.py
```

### Executing the Full Pipeline

To regenerate data, retrain the model, and run evaluation in one step:

```bash
python run_all.py
```

## Technical Stack

- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit

---

**ShopEase Sentiment Analysis Project**
# Shop-Ease
