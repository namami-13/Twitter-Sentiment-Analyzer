import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Fixed typo in 'extraction'
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Determine where to look for/save files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model.joblib')
TFIDF_PATH = os.path.join(SCRIPT_DIR, 'tfidf.joblib')

# 2. Try loading existing models
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        print("✅ Model loaded successfully from:", SCRIPT_DIR)
        return model, tfidf
    except FileNotFoundError:
        print("⚠️ Model files not found in:", SCRIPT_DIR)
        return None, None

# 3. Train new models if needed
def train_new_models():
    print("\nTraining new model...")
    # Sample training data (replace with your actual data)
    data = {
        'text': [
            "I love this product", "This is terrible",
            "Awesome experience", "Worst purchase ever",
            "Highly recommend", "Complete waste of money"
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    }
    
    # Create and train pipeline
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    model.fit(data['text'], data['label'])
    
    # Save components
    joblib.dump(model.named_steps['multinomialnb'], MODEL_PATH)
    joblib.dump(model.named_steps['tfidfvectorizer'], TFIDF_PATH)
    print(f"💾 Saved new model files to:\n{MODEL_PATH}\n{TFIDF_PATH}")
    return model.named_steps['multinomialnb'], model.named_steps['tfidfvectorizer']

# 4. Prediction function
def predict_sentiment(text, model, tfidf):
    text = re.sub(r'[^a-z\s]', '', text.lower()).strip()
    vec = tfidf.transform([text])
    proba = model.predict_proba(vec)[0]
    return {
        'prediction': ['negative', 'positive'][model.predict(vec)[0]],
        'confidence': {
            'negative': f"{proba[0]*100:.1f}%",
            'positive': f"{proba[1]*100:.1f}%"
        }
    }

# 5. Main execution
if __name__ == "__main__":
    print(f"🔍 Looking for model files in: {SCRIPT_DIR}")
    model, tfidf = load_models()
    
    if model is None:
        model, tfidf = train_new_models()
    
    # Test prediction
    while True:
        text = input("\nEnter text to analyze (or 'quit'): ")
        if text.lower() == 'quit':
            break
        result = predict_sentiment(text, model, tfidf)
        print(f"Result: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']}")

    print("\nTip: Keep these files for future use:")
    print(f"- {MODEL_PATH}")
    print(f"- {TFIDF_PATH}")

    #- C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\model.joblib
#C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\tfidf.joblib
