import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page config
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide"
)

# Custom CSS for Twitter-like UI
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-size: 18px !important;
        padding: 12px !important;
    }
    .stButton>button {
        background-color: #1DA1F2 !important;
        color: white !important;
        font-weight: bold !important;
        padding: 10px 24px !important;
    }
    .positive {
        color: #17BF63 !important;
        font-weight: bold;
    }
    .negative {
        color: #E0245E !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Twitter-specific preprocessing
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+|@(\w+)|#(\w+)', lambda m: m.group(1) or m.group(2) or '', tweet)
    tweet = re.sub(r'[^\w\s!?]', '', tweet)
    return ' '.join(tweet.split())

# Load model (replace with your actual paths)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'model.joblib' and 'tfidf.joblib' are in the same directory.")
        return None, None

model, vectorizer = load_model()

# App UI
st.title("🐦 Twitter Sentiment Analyzer")
st.subheader("Discover the emotion behind tweets")

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    tweet = st.text_input("Enter a tweet:", placeholder="Type or paste a tweet here...")
with col2:
    st.write("")  # Spacer
    analyze_btn = st.button("Analyze Sentiment")

# Sample tweets
sample_tweets = [
    "Just got the new iPhone - it's fire! 🔥",
    "@Airline your service ruined my vacation 😡",
    "LOL this is hilarious 😂",
    "Why does everything break so fast?!"
]

# Display sample tweets as clickable chips
st.write("Try sample tweets:")
cols = st.columns(len(sample_tweets))
for i, col in enumerate(cols):
    if col.button(sample_tweets[i][:30] + "..." if len(sample_tweets[i]) > 30 else sample_tweets[i]):
        tweet = sample_tweets[i]

# Analysis results
if analyze_btn or tweet:
    if not model:
        st.warning("Model not loaded - cannot analyze")
    else:
        cleaned = preprocess_tweet(tweet)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        prediction = model.predict(vec)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Tweet:**\n\n{tweet}")
        with col2:
            sentiment = "positive" if prediction == 1 else "negative"
            emoji = "😊" if sentiment == "positive" else "😠"
            st.markdown(f"""
            **Sentiment:** <span class="{sentiment}">{sentiment.upper()} {emoji}</span>
            
            **Confidence:**
            - Positive: {proba[1]*100:.1f}%
            - Negative: {proba[0]*100:.1f}%
            """, unsafe_allow_html=True)
        
        # Visual indicator
        st.progress(proba[1] if prediction == 1 else proba[0])
        st.caption("Sentiment strength")

# Footer
st.markdown("---")
st.caption("Note: This model works best with clear emotional language in tweets.")
