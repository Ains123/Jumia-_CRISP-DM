import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler


class BehavioralModel:
    """Predicts user purchase propensity with a focus on stability over skew."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric="logloss",
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Splits, balances (moderately), and fits the model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Conservative balancing: Bring minority up to 10% of majority
        ros = RandomOverSampler(sampling_strategy=0.1, random_state=self.random_state)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

        print(f"Balanced Training Distribution: {np.bincount(y_resampled)}")

        # Fit
        self.model.fit(X_resampled, y_resampled)

        # Call evaluate (now restored)
        self.evaluate(X_test, y_test)

        return self.model

    def evaluate(self, X_test, y_test):
        """Prints performance metrics."""
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        print("\n--- Behavioral Propensity Report ---")
        print(classification_report(y_test, preds))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")

    def save(self, model_path="../models/propensity_xgb.joblib"):
        """Persists the trained XGBoost model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"☑ Success: Propensity model saved to {model_path}")
