import joblib
import re

# Load your actual trained files from the .ipynb training
model = joblib.load("model.joblib")
tfidf = joblib.load("tfidf.joblib")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def predict_sentiment(text):
    text_clean = clean_text(text)
    if not text_clean:
        return {'prediction': 'unknown', 'confidence': {}}

    # Transform text using your saved TF-IDF vectorizer
    X = tfidf.transform([text_clean])

    # Predict using your trained model
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    confidence = {cls: f"{p * 100:.1f}%" for cls, p in zip(classes, proba)}

    return {'prediction': prediction, 'confidence': confidence}

# Example run
if __name__ == "__main__":
    print("✨ Sentiment Analyzer Ready! Using your trained model ✨")
    while True:
        text = input("\nEnter text (or 'quit' to exit): ").strip()
        if text.lower() == 'quit':
            print("👋 Exiting.")
            break
        result = predict_sentiment(text)
        print(f"\n🟢 Prediction: {result['prediction'].upper()}")
        print("📊 Confidence:")
        for cls, conf in result['confidence'].items():
            print(f"   {cls}: {conf}")
