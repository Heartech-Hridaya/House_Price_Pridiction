# House Price Prediction 🏠

A machine learning project that predicts residential house prices in India based on features like location, area, number of bathrooms, furnishing status, and more. Built end-to-end: data cleaning, EDA, feature engineering, model training, evaluation, and deployment as a live web app.

🔗 **Live app:** https://housepricepridictionbyhridaya.streamlit.app/

## Overview

The dataset contains ~187,000 real estate listings scraped from an Indian property site, with raw, messy fields (text-based prices like `"42 Lac"`, missing values, mixed units). The goal was to clean this data, explore it, engineer useful features, and train a regression model to predict total house price.

## Project structure

```
.
├── app.py                      # Streamlit web app
├── house_price_prediction.ipynb  # Full notebook: cleaning, EDA, training
├── house_price_model.pkl       # Trained Linear Regression model
├── location_price_map.pkl      # Location → average price mapping (for encoding)
├── model_columns.pkl           # Expected input column order for the model
├── requirements.txt            # Python dependencies
└── README.md
```

## Pipeline

1. **Data cleaning** — converted text-based prices (Lac/Cr) to numeric rupees, parsed `"X out of Y"` floor strings into separate columns, handled missing values using context-aware fills (e.g. location-grouped medians).
2. **EDA** — identified and removed outliers using the IQR method, analyzed correlations between features and price, found `Bathroom` and `Carpet_Area` as the strongest numeric predictors.
3. **Feature engineering** — target-encoded `location` (81 unique cities) by average price, one-hot encoded categorical fields (`Furnishing`, `Ownership`, `Transaction`).
4. **Model training** — trained a Linear Regression model as a baseline.
5. **Evaluation** — MAE, RMSE, and R² used to assess prediction quality.
6. **Deployment** — model pickled and served via a Streamlit app for interactive predictions.


## Tech stack

- **Python**, **pandas**, **scikit-learn** — data processing & modeling
- **Matplotlib / Seaborn** — visualization (in notebook)
- **Streamlit** — web app deployment

## Notes & limitations

- This is a learning project  — Linear Regression is used as a baseline; a tree-based model (e.g. Random Forest) would likely improve accuracy.
- `location` encoding uses average price across the full dataset rather than a strict train-only split, a simplification for this project's scope.
- Free-text fields (`Title`, `Description`) were excluded — not used for prediction.
