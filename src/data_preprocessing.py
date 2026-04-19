import re
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore")


def setup_nltk():
    """Ensure required NLTK components are available."""
    for res in ["punkt", "stopwords", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


setup_nltk()


class DataLoader:
    """Handles data ingestion from raw local storage."""

    def __init__(self, data_dir: str = "./data/raw/"):
        self.data_dir = data_dir

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        beh = pd.read_csv(f"{self.data_dir}ecommerce_user_behavior_8000.csv")
        prods = pd.read_csv(f"{self.data_dir}Products.csv")
        revs = pd.read_csv(f"{self.data_dir}Reviews.csv")
        return beh, prods, revs


class DataCleaner:
    """Cleans behavioral logs and handles missingness."""

    def __init__(self):
        self.numeric_cols = [
            "time_on_site",
            "pages_viewed",
            "previous_purchases",
            "cart_items",
            "avg_session_time",
            "bounce_rate",
        ]

    def clean_behavioral(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().dropna(subset=["user_id"])

        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        df["gender"] = df["gender"].fillna("Unknown")
        df["device_type"] = df["device_type"].fillna("Unknown")

        bool_cols = ["discount_seen", "ad_clicked", "returning_user", "purchase"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        return df


class TextPreprocessor:
    """Localized NLP processor for Kenyan slang and currency."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english")).union(
            {
                "niko",
                "uko",
                "yuko",
                "tuko",
                "mko",
                "wako",
                "hii",
                "hiyo",
                "huyo",
                "hawa",
                "hao",
                "sana",
                "tu",
            }
        )
        self.slang_map = {
            "feki": "fake",
            "mbaya": "bad",
            "chafu": "dirty",
            "poa": "good",
            "safi": "good",
            "noma": "great",
            "fiti": "good",
            "bamba": "good",
        }

    def parse_price(self, price_str: str) -> float:
        """Extracts float from 'KSh 1,234' format."""
        if pd.isna(price_str) or price_str == "":
            return np.nan
        clean_str = str(price_str).replace("KSh", "").replace(",", "").strip()
        if " - " in clean_str:
            clean_str = clean_str.split(" - ")[0]
        try:
            return float(clean_str)
        except ValueError:
            return np.nan

    def clean_text(self, text: str) -> str:
        """Lowercase, slang mapping, and regex noise removal."""
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        for slang, standard in self.slang_map.items():
            text = text.replace(slang, standard)
        text = re.sub(r"http\S+|www\S+|https\S+|[^\w\s]|\d+", " ", text)
        return " ".join(text.split())

    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return [
            self.stemmer.stem(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]

    def build_nlp_frame(
        self, reviews: pd.DataFrame, products: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges reviews and products on SKU -> Product_ID mapping."""
        products = products.copy()
        products["price_numeric"] = products["final_price"].apply(self.parse_price)

        # FIX: Mapping 'sku' from reviews to 'product_id' from products
        merged = reviews.merge(
            products[
                ["product_id", "product_name", "product_category", "price_numeric"]
            ],
            left_on="sku",
            right_on="product_id",
            how="left",
        )

        merged["sentiment_target"] = merged["rating"].apply(
            lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan)
        )

        merged["full_text"] = (
            merged["title"].fillna("") + " " + merged["review"].fillna("")
        ).apply(self.clean_text)
        merged["tokens"] = merged["full_text"].apply(self.tokenize)

        return merged[
            merged["sentiment_target"].notna() & (merged["full_text"] != "")
        ].copy()


class FeatureEngineer:
    """Feature generation for behavioral prediction."""

    def engineer(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df["cart_depth"] = df["cart_items"] / df["pages_viewed"].replace(0, 1)
        df["engaged"] = (df["pages_viewed"] > df["pages_viewed"].median()).astype(int)

        feats = [
            "age",
            "time_on_site",
            "pages_viewed",
            "previous_purchases",
            "cart_items",
            "avg_session_time",
            "bounce_rate",
            "cart_depth",
            "engaged",
            "discount_seen",
            "ad_clicked",
        ]

        return df[feats].fillna(0), df["purchase"]


if __name__ == "__main__":
    loader = DataLoader()
    beh, prods, revs = loader.load_all()

    cleaner = DataCleaner()
    text_proc = TextPreprocessor()
    feat_eng = FeatureEngineer()

    # Behavioral Pipeline
    X, y = feat_eng.engineer(cleaner.clean_behavioral(beh))
    print(f"✅ Behavioral Stream: {X.shape}")

    # NLP Pipeline
    nlp_df = text_proc.build_nlp_frame(revs, prods)
    print(f"✅ NLP Stream: {nlp_df.shape}")
