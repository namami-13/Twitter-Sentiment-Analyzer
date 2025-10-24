# Twitter Sentiment Analyzer 🐦🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-75%25-brightgreen)

A machine learning project that classifies tweet sentiment using Python and Scikit-Learn.

## 📂 Project Structure
TwitterSentiment/<br>
├── model.ipynb       # Jupyter notebook for model development<br>
├── twitter_sentiment_app.py       # Streamlit web interface<br>
├── sentiment_app.py     # Core prediction logic<br>
├── requirements.txt     # Python dependencies<br>
├── data/<br>
│ └── tweets.csv     # Sample dataset <br>
└── models/<br>
├── model.joblib     # Trained classifier<br>
└── tfidf.joblib     # TF-IDF vectorizer<br>

---

## 🧹 Data Preprocessing
### Text Cleaning Function
```python
import re
from nltk.corpus import stopwords

def clean_tweet(tweet):
    # Remove URLs and mentions
    tweet = re.sub(r'http\S+|@\w+', '', tweet)
    # Keep only letters and spaces
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove stopwords
    stops = set(stopwords.words('english'))
    return ' '.join([word for word in tweet.split() if word not in stops])

```
## 🤖 Model Development
### Feature Extraction (TF-IDF)

```Python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=15000,
    stop_words='ENGLISH_STOP_WORDS'
)
X = tfidf.fit_transform(cleaned_tweets)
```
> TF-IDF transforms raw text into a vector of numbers that reflect the importance of each word.

## 📊 Evaluation Metrics
### Classification Report
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Negative   | 0.77      | 0.83   | 0.79     | 266     |
| Neutral    | 0.85      | 0.81   | 0.83     | 457     |
| Positive   | 0.83      | 0.82   | 0.82     | 277     |
|            |           |        |          |         |
| **Accuracy** |          |        | **0.82** | 1000    |
| **Macro Avg** | 0.81    | 0.82   | 0.82     | 1000    |
| **Weighted Avg** | 0.82 | 0.82   | 0.82     | 1000    |

Key:
- **Precision**: Percentage of correct positive predictions
- **Recall**: Percentage of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall<br>

> These metrics reflect the model's performance on the test set for sentiment classification.

## 🚀 How to Use
1. Install dependencies:
```Bash
pip install -r requirements.txt
```
2. Run the Streamlit app:
```Bash
streamlit run twitter_sentiment_app.py
```
3. To retrain models:
  * Open model.ipynb in Jupyter
  * Run all cells
  * New models will save to models/







