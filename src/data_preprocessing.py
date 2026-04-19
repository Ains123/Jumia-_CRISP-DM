import re
import warnings
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
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
        """Handles both KSh and EGP currency formats."""
        if pd.isna(price_str) or price_str == "":
            return np.nan

        # Remove currency symbols and formatting
        clean_str = (
            str(price_str)
            .replace("KSh", "")
            .replace("EGP", "")
            .replace(",", "")
            .strip()
        )

        # Handle ranges: "EGP 329.99 - EGP 399.99" -> 329.99
        if " - " in clean_str:
            clean_str = clean_str.split(" - ")[0].replace("EGP", "").strip()

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
        self, reviews: pd.DataFrame, products: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Builds NLP-ready dataframe from reviews.

        NOTE: products parameter is accepted for API compatibility but NOT merged.
        SKUs in reviews do not match product_id in products.csv (0% overlap).
        Sentiment analysis is performed on reviews alone.

        Args:
            reviews: DataFrame with columns 'rating', 'title', 'review', 'sku'
            products: Optional DataFrame (ignored for merging, kept for API compatibility)

        Returns:
            DataFrame with columns: sentiment_target, full_text, tokens, product_name, etc.
        """
        df = reviews.copy()

        # Optional: parse prices if products provided (for reference only, not merged)
        if products is not None:
            products = products.copy()
            products["price_numeric"] = products["final_price"].apply(self.parse_price)
            print(
                "⚠️ Note: SKU mismatch detected. Running sentiment on reviews only (no product enrichment)."
            )

        # Create sentiment target from ratings
        # 4-5 stars = positive (1), 1-2 stars = negative (0), 3 stars = neutral (drop)
        df["sentiment_target"] = df["rating"].apply(
            lambda x: 1 if x >= 4 else (0 if x <= 2 else np.nan)
        )

        # Combine title and review for text analysis
        df["full_text"] = (
            df["title"].fillna("") + " " + df["review"].fillna("")
        ).apply(self.clean_text)
        df["tokens"] = df["full_text"].apply(self.tokenize)

        # Add placeholder columns for API compatibility (no merge attempted)
        df["product_name"] = None
        df["product_category"] = None
        df["price_numeric"] = None

        # Filter out neutral ratings (3 stars) and empty text
        result = df[df["sentiment_target"].notna() & (df["full_text"] != "")].copy()

        print(f"✅ Built NLP frame: {len(result)} reviews processed")
        print(f"   Positive (4-5 stars): {sum(result['sentiment_target'] == 1)}")
        print(f"   Negative (1-2 stars): {sum(result['sentiment_target'] == 0)}")

        return result


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

    # NLP Pipeline (no product merge attempted)
    nlp_df = text_proc.build_nlp_frame(revs, prods)
    print(f"✅ NLP Stream: {nlp_df.shape}")
