import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataLoader:
    """Class for loading e-commerce datasets."""

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.behavioral_data = None
        self.products_data = None
        self.reviews_data = None

    def load_behavioral_data(
        self, filename: str = "ecommerce_user_behavior_8000.csv"
    ) -> pd.DataFrame:
        """Load user session behavioral data."""
        filepath = f"{self.data_dir}raw/{filename}"
        self.behavioral_data = pd.read_csv(filepath)
        print(f"Loaded behavioral data: {self.behavioral_data.shape}")
        return self.behavioral_data

    def load_product_data(self, filename: str = "Products.csv") -> pd.DataFrame:
        """Load product catalog data."""
        filepath = f"{self.data_dir}raw/{filename}"
        self.products_data = pd.read_csv(filepath)
        print(f"Loaded product data: {self.products_data.shape}")
        return self.products_data

    def load_reviews_data(self, filename: str = "Reviews.csv") -> pd.DataFrame:
        """Load customer reviews data."""
        filepath = f"{self.data_dir}raw/{filename}"
        self.reviews_data = pd.read_csv(filepath)
        print(f"Loaded reviews data: {self.reviews_data.shape}")
        return self.reviews_data

    def quick_summary(self) -> dict:
        """Get quick summary of all datasets."""
        summary = {}

        if self.behavioral_data is not None:
            summary["behavioral"] = {
                "shape": self.behavioral_data.shape,
                "columns": self.behavioral_data.columns.tolist(),
                "missing": self.behavioral_data.isnull().sum().sum(),
            }

        if self.products_data is not None:
            summary["products"] = {
                "shape": self.products_data.shape,
                "columns": self.products_data.columns.tolist(),
            }

        if self.reviews_data is not None:
            summary["reviews"] = {
                "shape": self.reviews_data.shape,
                "columns": self.reviews_data.columns.tolist(),
            }

        return summary


if __name__ == "__main__":
    # Test the loader
    loader = DataLoader()
    loader.load_behavioral_data()
    loader.load_product_data()
    loader.load_reviews_data()

    summary = loader.quick_summary()
    print("\n=== Data Summary ===")
    for name, info in summary.items():
        print(f"\n{name.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
