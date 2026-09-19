import pandas as pd
import numpy as np
from src.config import *

FEATURES = [
    "lag_1","lag_7","lag_14","lag_28",
    "rolling_mean_7","rolling_mean_14","rolling_mean_28",
    "rolling_std_7","rolling_std_28",
    "day_of_week","month","quarter","is_weekend",
    "promotion","is_holiday","current_stock","on_order",
    "lead_time_days","safety_stock","reorder_point"
]

def main():
    path = PROCESSED_DIR / "analysis_ready.csv"
    if not path.exists():
        from src.pipeline import build_analysis_ready
        build_analysis_ready()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["sku_id","date"]).copy()

    # Past-only lags.
    g = df.groupby("sku_id", group_keys=False)
    df["lag_1"] = g["units_sold"].shift(1)
    df["lag_7"] = g["units_sold"].shift(7)
    df["lag_14"] = g["units_sold"].shift(14)
    df["lag_28"] = g["units_sold"].shift(28)

    # IMPORTANT: shift before rolling = no use of today's target.
    past = g["units_sold"].shift(1)
    df["rolling_mean_7"] = past.groupby(df["sku_id"]).transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["rolling_mean_14"] = past.groupby(df["sku_id"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
    df["rolling_mean_28"] = past.groupby(df["sku_id"]).transform(lambda s: s.rolling(28, min_periods=14).mean())
    df["rolling_std_7"] = past.groupby(df["sku_id"]).transform(lambda s: s.rolling(7, min_periods=3).std())
    df["rolling_std_28"] = past.groupby(df["sku_id"]).transform(lambda s: s.rolling(28, min_periods=14).std())

    # Ensure expected columns exist.
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0

    model_df = df.dropna(subset=["lag_28","rolling_mean_28"]).copy()
    model_df[FEATURES] = model_df[FEATURES].replace([np.inf,-np.inf], np.nan)
    model_df[FEATURES] = model_df[FEATURES].fillna(0)

    model_df.to_csv(PROCESSED_DIR / "model_features.csv", index=False)
    print(f"Saved {len(model_df):,} rows to {PROCESSED_DIR / 'model_features.csv'}")

if __name__ == "__main__":
    main()
