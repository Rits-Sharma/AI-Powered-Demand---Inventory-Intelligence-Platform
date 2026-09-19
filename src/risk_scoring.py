import numpy as np
import pandas as pd
from src.config import *

def main():
    pred_path = PROCESSED_DIR / "test_predictions.csv"
    if not pred_path.exists():
        from src.forecast_model import main as train_model
        train_model()

    pred = pd.read_csv(pred_path, parse_dates=["date"])
    ready = pd.read_csv(PROCESSED_DIR / "analysis_ready.csv", parse_dates=["date"])

    # Use latest available inventory record per SKU.
    inv_cols = ["sku_id","date","current_stock","on_order","lead_time_days","safety_stock","reorder_point","inventory_value","category","product_name"]
    available = [c for c in inv_cols if c in ready.columns]
    inv = ready[available].sort_values(["sku_id","date"]).drop_duplicates("sku_id", keep="last")

    # Forecast average daily demand from the last 28 days of model predictions.
    recent = pred.groupby("sku_id").tail(28)
    fc = recent.groupby("sku_id").agg(
        avg_forecast_daily=("xgboost_pred","mean"),
        forecast_units_28d=("xgboost_pred","sum")
    ).reset_index()

    risk = inv.merge(fc, on="sku_id", how="left")
    risk["avg_forecast_daily"] = risk["avg_forecast_daily"].fillna(0)
    risk["days_of_cover"] = np.where(
        risk["avg_forecast_daily"] > 0,
        risk["current_stock"].fillna(0) / risk["avg_forecast_daily"],
        np.inf
    )

    risk["inventory_vs_reorder_point"] = risk["current_stock"].fillna(0) - risk["reorder_point"].fillna(0)
    risk["stockout_risk"] = (
        (risk["days_of_cover"] <= risk["lead_time_days"].fillna(0).clip(lower=0)) |
        (risk["current_stock"].fillna(0) <= risk["safety_stock"].fillna(0))
    )
    risk["overstock_risk"] = (
        (risk["days_of_cover"] >= 60) &
        (risk["current_stock"].fillna(0) > risk["reorder_point"].fillna(0))
    )

    def classify(r):
        if r["stockout_risk"]:
            return "Reorder Now"
        if r["overstock_risk"]:
            return "Markdown/Clear"
        if r["days_of_cover"] < 14:
            return "Watch/Volatile"
        return "Healthy"

    risk["risk_category"] = risk.apply(classify, axis=1)

    risk.to_csv(PROCESSED_DIR / "risk_scores.csv", index=False)

    summary = risk["risk_category"].value_counts().to_dict()
    import json
    (REPORTS_DIR / "risk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Risk summary:", summary)

if __name__ == "__main__":
    main()
