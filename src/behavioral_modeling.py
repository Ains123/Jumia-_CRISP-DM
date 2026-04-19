import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# We switch to the specialized imbalanced-learn forest
from imblearn.ensemble import BalancedRandomForestClassifier


class BehavioralModel:
    """A High-Sensitivity model designed to find signal in near-zero recall scenarios."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        # This model balances every bootstrap sample automatically
        self.model = BalancedRandomForestClassifier(
            n_estimators=500,
            sampling_strategy="all",  # Force 50/50 balance in every tree
            replacement=True,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        print(f"Original Training Distribution: {np.bincount(y_train)}")
        print("Training with Balanced Forest (Internal Re-sampling)...")

        # We don't need manual SMOTE here; the forest handles it internally per tree
        self.model.fit(X_train, y_train)

        self.evaluate(X_test, y_test)

        # FEATURE IMPORTANCE CHECK
        # Let's see if ANY feature is actually helping
        importances = pd.Series(self.model.feature_importances_, index=X.columns)
        print("\n--- Top Predictive Signals ---")
        print(importances.sort_values(ascending=False).head(5))

        return self.model

    def evaluate(self, X_test, y_test):
        # We use a lower threshold (0.4) to catch more Class 0
        probs = self.model.predict_proba(X_test)[:, 0]  # Prob of being Class 0
        preds = (self.model.predict_proba(X_test)[:, 1] > 0.4).astype(int)

        print("\n--- Behavioral Propensity Report (Threshold Adjusted) ---")
        print(classification_report(y_test, preds))
        print(
            f"ROC-AUC Score: {roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]):.4f}"
        )

        print("\n--- Confusion Matrix (Actual vs Predicted) ---")
        print(confusion_matrix(y_test, preds))

    def save(self, model_path="../models/propensity_brf.joblib"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"☑ Success: High-Sensitivity model saved to {model_path}")
