import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Visualizer:
    def __init__(self, style="seaborn-v0_8-muted"):
        plt.style.use(style)
        self.fig_dir = "../reports/figures/"

    def plot_feature_importance(self, model, feature_names, title="Feature Importance"):
        """Visualizes which behavioral signals are strongest."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title(title)
        plt.xlabel("Relative Importance Score")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}behavioral_importance.png")
        plt.show()

    def plot_intervention_matrix(self, df):
        """
        Creates a 2x2 Matrix:
        X-axis: Sentiment (0=Bad, 1=Good)
        Y-axis: Purchase Propensity (0=Leaving, 1=Staying)
        """
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            data=df,
            x="predicted_sentiment",
            y="propensity_score",  # The raw probability from the BRF
            hue="intervention_category",
            palette="viridis",
            alpha=0.6,
        )
        plt.axhline(0.4, ls="--", color="red", alpha=0.5)  # Our threshold
        plt.axvline(0.5, ls="--", color="red", alpha=0.5)
        plt.title("Customer Intervention Matrix")
        plt.savefig(f"{self.fig_dir}intervention_matrix.png")
        plt.show()
