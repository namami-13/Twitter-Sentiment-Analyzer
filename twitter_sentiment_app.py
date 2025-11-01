import streamlit as st
import joblib
import re
import os
import numpy as np

# ✅ --- Page Config ---
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide"
)

# ✅ --- Custom CSS (Twitter-style) ---
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 18px !important;
        padding: 12px !important;
    }
    .stButton > button {
        background-color: #1DA1F2 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 24px !important;
        border-radius: 6px !important;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #0d8ddb !important;
    }
    .positive { color: #17BF63 !important; font-weight: bold; }
    .negative { color: #E0245E !important; font-weight: bold; }
    .neutral  { color: #8899A6 !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ✅ --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

# ✅ --- Load your trained model & vectorizer ---
@st.cache_resource
def load_model_files():
    model_path = "model.joblib"
    tfidf_path = "tfidf.joblib"

    if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
        st.error("⚠️ Model or TF-IDF files not found! Please make sure both 'model.joblib' and 'tfidf.joblib' are in the same folder as this app.")
        return None, None

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    st.success("✅ Trained model and vectorizer loaded successfully!")
    return model, tfidf

model, tfidf = load_model_files()

# ✅ --- Header Section ---
st.title("🐦 Twitter Sentiment Analyzer")
st.subheader("Analyze tweet emotions using your trained machine learning model (82% accuracy)")

# ✅ --- Input Section ---
col1, col2 = st.columns([3, 1])
with col1:
    tweet = st.text_input("Enter a tweet:", placeholder="Type or paste a tweet here...")
with col2:
    st.write("")  # spacer
    analyze_btn = st.button("Analyze Sentiment")

# ✅ --- Sample Tweets ---
st.markdown("**Try these sample tweets:**")
sample_tweets = [
    "Just got the new iPhone - it's fire! 🔥",
    "@Airline your service ruined my vacation 😡",
    "LOL this is hilarious 😂",
    "Why does everything break so fast?!",
    "The concert last night was amazing 🎶",
    "This product is not worth the money 😤",
    "Feeling blessed and grateful today 🙏"
]

cols = st.columns(len(sample_tweets))
for i, col in enumerate(cols):
    if col.button(sample_tweets[i][:25] + "..." if len(sample_tweets[i]) > 25 else sample_tweets[i]):
        tweet = sample_tweets[i]

# ✅ --- Prediction Section ---
if analyze_btn and tweet:
    if model is None or tfidf is None:
        st.warning("⚠️ Model not loaded — cannot analyze sentiment.")
    else:
        cleaned_text = clean_text(tweet)
        X = tfidf.transform([cleaned_text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        # --- Display Results ---
        st.markdown("---")
        st.subheader("🧠 Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Tweet:**\n\n> {tweet}")

        with col2:
            emoji_dict = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}
            emoji = emoji_dict.get(prediction, '')

            confidence_text = "\n".join(
                [f"- **{cls.capitalize()}**: {p * 100:.1f}%" for cls, p in zip(classes, proba)]
            )

            st.markdown(f"""
            **Sentiment:** <span class="{prediction}">{prediction.upper()} {emoji}</span>

            **Confidence:**
            {confidence_text}
            """, unsafe_allow_html=True)

        st.progress(float(np.max(proba)))
        st.caption("Confidence represents how sure the model is about its prediction.")

# ✅ --- Footer ---
st.markdown("---")
st.caption("⚙️ *Note: This Streamlit app uses your trained model (`model.joblib` & `tfidf.joblib`) for real predictions.*")
