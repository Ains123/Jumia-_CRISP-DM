# Integrated Sentiment & Behavioral Analytics for Kenyan E-Commerce

## Project Overview
This project analyzes customer behavior and sentiment for a Kenyan e-commerce platform, combining:
- **NLP Sentiment Analysis**: Classify Jumia reviews as Satisfied/Unsatisfied
- **Predictive Modeling**: Predict purchase conversion from session behavior
- **Model Interpretability**: Explain predictions using SHAP and LIME

## Business Problem
Local vendors face two key challenges:
1. High traffic volume with low conversion rates
2. Rating-sentiment gaps where star ratings don't reflect actual complaints

## Dataset
- **Behavioral Data**: 8,000 user sessions with 14 features
- **Product Data**: Jumia product listings with prices (KSh)
- **Reviews Data**: Customer reviews with ratings and text

## Project Structure
```
ecommerce-sentiment-analysis/
├── data/
│   ├── raw/
│   │   ├── ecommerce_user_behavior_8000.csv
│   │   ├── Products.csv
│   │   └── Reviews.csv
│   └── processed/
│       └── (cleaned datasets will be saved here)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_modeling.ipynb
│   ├── 03_behavioral_modeling.ipynb
│   └── 04_interpretability.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── sentiment_analysis.py
│   ├── behavioral_modeling.py
│   └── visualization.py
├── models/
│   └── (saved model files)
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
├── .gitignore
└── presentation.pdf
```

## Installation

```bash
# Clone repository
git clone https://github.com/Ains123/Jumia-_CRISP-DM.git
cd ecommerce-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Authors
Lorenah -M: Lead Developer (Full Pipeline)

Ainsley -W : Data Quality & Testing

Angela -M: Documentation & Presentation

Dennis -K: Visualization & Dashboard

## License
MIT Licensed