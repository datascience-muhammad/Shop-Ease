from extras.generate_synthetic_data import generate_reviews
from src.preprocessing import preprocess_dataframe
from src.train import train_bert
from src.evaluate import evaluate
from src.model_loader import load_model
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from sklearn.model_selection import train_test_split
import os

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    print("\n" + "="*50)
    print("🚀 STARTING SHOP-EASE END-TO-END PIPELINE")
    print("="*50 + "\n")

    # 1. Generate synthetic data
    print("[1/5] Generating synthetic data...")
    df_raw = generate_reviews(1000)
    save_csv(df_raw, RAW_DATA_PATH)

    # 2. Preprocess
    print("[2/5] Preprocessing and cleaning data...")
    df_clean = preprocess_dataframe(df_raw)
    save_csv(df_clean, PROCESSED_DATA_PATH)

    # 3. Train-Test Split
    print("[3/5] Splitting data for training and testing...")
    X = df_clean['final_text'].astype(str)
    y = df_clean['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train
    print("[4/5] Training BERT model (this may take a few minutes)...")
    train_bert(X_train.tolist(), y_train.tolist())

    # 5. Evaluate
    print("[5/5] Evaluating final model...")
    tokenizer, model = load_model()
    evaluate(model, X_test.tolist(), y_test.tolist(), tokenizer)

    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE - DASHBOARD READY")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
