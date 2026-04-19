import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def identity_fn(x):
    """Identity function to bypass TfidfVectorizer's default tokenization."""
    return x


class SentimentModel:
    """Classifies consumer sentiment using localized text features."""

    def __init__(self, max_features=2500):
        # Using the named function identity_fn instead of a lambda
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            tokenizer=identity_fn,
            preprocessor=identity_fn,
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )

    def prepare_features(self, tokens_series: pd.Series):
        return self.vectorizer.fit_transform(tokens_series)

    def train(self, X, y):
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
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vec_path)
        print(f"☑ Success: Sentiment components saved to {model_path} and {vec_path}")
