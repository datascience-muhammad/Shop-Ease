import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Fix path to allow imports from src
root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.predict import predict_sentiment
from src.config import BERT_MODEL_NAME

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ShopEase AI",
    layout="wide",
)

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; background-color: #0d6efd; color: white; border: none; }
    .stButton>button:hover { background-color: #0b5ed7; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: TEST EXAMPLES ---
with st.sidebar:
    st.header("Test Examples")
    st.markdown("Select a sample to populate the input:")
    
    st.subheader("Positive / Negative")
    if st.button("Perfect Product"):
        st.session_state.text_input = "The product arrived on time and works perfectly! 5 stars."
    if st.button("Terrible Service"):
        st.session_state.text_input = "Terrible service, it arrived broken. Useless!"
        
    st.subheader("Nuanced / Hard")
    if st.button("Mixed Feedback"):
        st.session_state.text_input = "It's not bad, but I expected more features for the price."

    st.divider()
    st.caption("Resets text area on click.")

# --- HEADER ---
st.title("ShopEase Sentiment Analysis")
st.text(f"Model: {BERT_MODEL_NAME} | Status: Educational Demo")

# Subtle warning instead of a big box
st.caption("⚠️ Note: This model is trained on a limited subset (100 samples). Predictions may be biased due to class imbalance.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Single Review")
    
    # Handle session state for sidebar buttons
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
        
    user_input = st.text_area("Customer Feedback", value=st.session_state.text_input, height=150)
    
    if st.button("Analyze", key="single"):
        if user_input.strip():
            with st.spinner("Processing..."):
                label, confidence = predict_sentiment(user_input)
                
                # Minimalist result display
                st.markdown("---")
                st.markdown(f"#### Sentiment: **{label}**")
                st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                
                if confidence < 0.6:
                    st.caption("Low confidence prediction.")
        else:
            st.warning("Please enter text.")

with col2:
    st.subheader("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "review" in df.columns:
            if st.button("Process Batch", key="batch"):
                with st.spinner("Processing..."):
                    results = [predict_sentiment(text) for text in df["review"]]
                    df["sentiment"], df["confidence"] = zip(*results)
                    
                    st.success("Analysis complete")
                    st.dataframe(df.head(50), use_container_width=True)
                    
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )
        else:
            st.error("CSV must contain a 'review' column.")

st.divider()
st.caption("ShopEase AI • Internal Tool")
