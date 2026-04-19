import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class SentimentModel:
    """Classifies consumer sentiment using localized text features."""

    def __init__(self, max_features=2500):
        # We use Tfidf with n-grams (1,2) to catch phrases like "not good" or "feki sana"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            tokenizer=lambda x: x,  # Already tokenized in preprocessing
            preprocessor=lambda x: x,
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )

    def prepare_features(self, tokens_series: pd.Series):
        """Transforms preprocessed tokens into a TF-IDF matrix."""
        return self.vectorizer.fit_transform(tokens_series)

    def train(self, X, y):
        """Fits the Random Forest classifier to the vectorized data."""
        self.classifier.fit(X, y)
        return self.classifier

    def evaluate(self, X_test, y_test):
        preds = self.classifier.predict(X_test)
        print("--- Sentiment Engine: Detailed Report ---")
        print(classification_report(y_test, preds))
        return confusion_matrix(y_test, preds)

    def save(
        self,
        model_path="../models/sentiment_rf.joblib",
        vec_path="../models/tfidf_vec.joblib",
    ):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vec_path)
        print(f"Sentiment components saved to {model_path} and {vec_path}")
