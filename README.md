# Project FORESIGHT

AI-powered demand forecasting and inventory intelligence platform.

## What this version contains

1. Data loading + cleaning for the four raw extracts
2. EDA with labelled charts and automated insight report
3. Leakage-free feature engineering
4. Seasonal-naive baseline
5. XGBoost forecasting model only
6. Chronological 8-week holdout evaluation
7. Inventory risk scoring
8. Streamlit dashboard
9. Saved model + metrics + reports

## Expected raw files

Put these four files inside `data/raw/`:

- `sales_daily.csv`
- `sku_master.csv`
- `calendar.csv`
- `inventory_snapshots.csv`

Expected columns are handled flexibly, including common variants such as:

### sales_daily
`Date, SKU, Units_Sold, Revenue, Price, Promotion`

### sku_master
`SKU, Product_Name, Category, Subcategory, Launch_Date, Cost_Price, Selling_Price, Gross_Margin_Per_Unit`

### calendar
`date, year, month, quarter, week, day_of_week, is_weekend, season, holiday, is_holiday, promotion_event`

### inventory_snapshots
`Snapshot_Date, SKU, Current_Stock, On_Order, Lead_Time_Days, Safety_Stock, Reorder_Point, Inventory_Value`

## Windows setup

Open a terminal in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run everything

```bash
python run_pipeline.py
```

Then launch the dashboard:

```bash
streamlit run app/dashboard.py
```

## Individual steps

```bash
python src/pipeline.py
python src/eda.py
python src/features.py
python src/baseline.py
python src/forecast_model.py
python src/risk_scoring.py
```

## Important modeling choices

- XGBoost is the only ML forecasting model.
- Train/test split is chronological.
- The final 8 weeks are held out for evaluation.
- Lag and rolling features use past observations only.
- Rolling features are shifted before calculation to prevent leakage.
- Seasonal-naive baseline uses the value from 7 days earlier.
- WAPE is reported alongside MAE and RMSE.

## Outputs

- `data/processed/analysis_ready.csv`
- `data/processed/model_features.csv`
- `data/processed/test_predictions.csv`
- `data/processed/risk_scores.csv`
- `models/xgboost_demand_model.joblib`
- `reports/model_metrics.json`
- `reports/eda_insights.json`
- `reports/figures/*.png`

## If you already have cleaned data

You can still run the full pipeline. The cleaning step is designed to be idempotent and will normalize column names, dates, duplicates, invalid sales values, and joins.
