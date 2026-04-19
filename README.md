
# E-Commerce Intelligence: Dual-Analysis Framework
## Overview
This project provides two **independent** analytical frameworks for e-commerce optimization in the Kenyan market:
1. **Purchase Propensity Engine** - Predicts user purchase likelihood from session behavior
2. **Sentiment Classifier** - Detects negative review sentiment with Sheng/Swahili slang support
> ⚠️ **Important Note**: These analyses are independent. The behavioral and review datasets cannot be linked due to mismatched identifiers (SKU vs product_id). Each model stands alone.
## Problem Statement
E-commerce platforms face two distinct challenges:
- **Behavioral**: Identifying high-intent users before they leave (99:1 class imbalance)
- **Qualitative**: Detecting dissatisfaction hidden in local slang within 5-star reviews
## Solution Architecture
```
┌─────────────────────────────────────────────────────────────────┐  
│ DATA SOURCES │  
├─────────────────────┬───────────────────┬───────────────────────┤  
│ Behavioral Data │ Product Catalog │ Reviews │  
│ (session logs) │ (Egypt/Kenya) │ (Kenyan customers) │  
└──────────┬──────────┴─────────┬─────────┴───────────┬───────────┘  
│ │ │  
▼ ▼ ▼  
┌─────────────────────┐ ┌─────────────────────────────────────────┐  
│ Purchase │ │ Sentiment Classification │  
│ Propensity │ │ (Independent - No Merge) │  
│ │ │ │  
│ • Balanced Random │ │ • TF-IDF + Random Forest │  
│ Forest │ │ • Sheng/Swahili slang mapping │  
│ • SMOTE handling │ │ • Information asymmetry detection │  
│ • 99:1 imbalance │ │ │  
└──────────┬──────────┘ └───────────────────┬─────────────────────┘  
│ │  
▼ ▼  
┌─────────────────────┐ ┌─────────────────────────────────────────┐  
│ Output: │ │ Output: │  
│ Churn probability │ │ Sentiment score + asymmetry flags │  
│ per session │ │ per review │  
└─────────────────────┘ └─────────────────────────────────────────┘

```

## Datasets
| Dataset | Records | Purpose | Key Columns |
|---------|---------|---------|-------------|
| `ecommerce_user_behavior_8000.csv` | 8,000 sessions | Purchase prediction | time_on_site, bounce_rate, cart_items, purchase |
| `Products.csv` | 100 products | Product metadata (unused in merge) | product_id, final_price |
| `Reviews.csv` | 100 reviews | Sentiment training | rating, review, title |
## Model Performance
### Behavioral Model (Purchase Propensity)
| Metric | Value |
|--------|-------|
| Class imbalance | 99:1 (non-purchase:purchase) |
| Algorithm | Balanced Random Forest |
| Top predictors | bounce_rate, time_on_site, avg_session_time |
| ROC-AUC | ~0.57 |
### Sentiment Model (Review Classification)
| Metric | Value |
|--------|-------|
| Training samples | 92 reviews |
| Classes | Positive (4-5⭐), Negative (1-2⭐) |
| Algorithm | Random Forest + TF-IDF |
| Slang support | Sheng/Swahili mapping |
## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ecommerce-intelligence.git
cd ecommerce-intelligence
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Download NLTK data (auto-downloads on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

## Usage

Run notebooks in order:
```
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_data_exploration.ipynb
# 2. Train sentiment model (reviews only)
jupyter notebook notebooks/02_sentiment_modeling.ipynb
# 3. Train behavioral model (session data only)
jupyter notebook notebooks/03_behavioral_modeling.ipynb
# 4. View final summary
jupyter notebook notebooks/06_final_summary.ipynb
```

### Command-line execution

```

# Run all analyses
jupyter nbconvert --to notebook --execute notebooks/0*.ipynb
# Or run specific model training
python -c "from src.behavioral_modeling import BehavioralModel; \
 from src.data_preprocessing import DataLoader, DataCleaner, FeatureEngineer; \
 loader = DataLoader(); beh, _, _ = loader.load_all(); \
 cleaner = DataCleaner(); feat = FeatureEngineer(); \
 X, y = feat.engineer(cleaner.clean_behavioral(beh)); \
 model = BehavioralModel(); model.train(X, y)"


## Project Structure
```
ecommerce-intelligence/
├── data/
│   └── raw/                          # Original datasets
│       ├── ecommerce_user_behavior_8000.csv
│       ├── Products.csv
│       └── Reviews.csv
├── models/                           # Trained models
│   ├── propensity_brf.joblib         # Behavioral model
│   ├── sentiment_rf.joblib           # Sentiment model
│   └── tfidf_vec.joblib              # TF-IDF vectorizer
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA
│   ├── 02_sentiment_modeling.ipynb   # Sentiment training
│   ├── 03_behavioral_modeling.ipynb  # Behavioral training
│   └── 06_final_summary.ipynb        # Summary & recommendations
├── reports/
│   └── figures/                      # Generated visualizations
├── src/
│   ├── data_preprocessing.py         # Loaders, cleaners, tokenizers
│   ├── behavioral_modeling.py        # BRF purchase predictor
│   ├── sentiment_analysis.py         # Slang-aware sentiment
│   └── visualization.py              # Plotting utilities
├── requirements.txt
└── README.md
```

## Key Findings

### Behavioral Analysis

-   Purchase events represent only **2-3%** of sessions
    
-   **Bounce rate** and **time on site** are strongest purchase predictors
    
-   Balanced Random Forest successfully identifies at-risk sessions despite 99:1 imbalance
    

### Sentiment Analysis

-   **85%** of reviews are 4-5 stars (typical e-commerce positivity bias)
    
-   Slang mapping (`feki` → `fake`, `mbaya` → `bad`) improves detection accuracy
    
-   ~7% of high-rated reviews contain negative sentiment patterns
    

### Data Limitations (Documented)

-   ❌ No join key between behavioral and review datasets
    
-   ❌ SKU field in reviews does not match product_id in catalog
    
-   ❌ Cross-analysis (e.g., "do negative reviewers purchase less?") is impossible
    
-   ✅ Each analysis validated independently
    

## Recommendations

### For Behavioral Team

1.  Deploy BRF model to flag high-churn sessions in real-time
    
2.  Focus retention on users with `bounce_rate > 0.8` and `time_on_site < 10`
    
3.  A/B test personalized offers for predicted non-purchasers
    

### For Product/Review Team

1.  Manually review identified information asymmetry cases
    
2.  Add Sheng/Swahili slang to standard monitoring lexicons
    
3.  Monitor negative tokens (`feki`, `mbaya`) in review streams
    

### For Data Engineering

1.  Implement consistent product identifiers across all data sources
    
2.  Add `user_id` to review records for user-level linking
    
3.  Consider data warehouse with foreign key constraints
    

## Requirements

    pandas>=2.0.0
    numpy>=1.24.0
    scikit-learn>=1.3.0
    imbalanced-learn>=0.11.0
    nltk>=3.8.0
    matplotlib>=3.7.0
    seaborn>=0.12.0
    joblib>=1.3.0
    jupyter>=1.0.0

## Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No join keys between datasets | Cross-analysis impossible | Treat as separate analyses |
| Small review sample (n=92) | Limited generalizability | Validate with larger dataset |
| 99:1 class imbalance | High precision, lower recall | BRF with SMOTE handling |
| Egyptian product catalog | Not applicable to Kenya | Remove cross-market inference |


## License

MIT

## Contributors

-   Lorenah -M, Ainsley -G, Angela -M, Dennis -K.

    

## Acknowledgments

-   NLTK for stopwords and tokenization
    
-   Scikit-learn for ML implementations
    
-   Imbalanced-learn for BRF algorithm
