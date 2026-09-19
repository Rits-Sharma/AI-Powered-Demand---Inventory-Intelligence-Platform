import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def clean_col(c):
    c = str(c).strip()
    c = re.sub(r"[^0-9a-zA-Z]+", "_", c)
    return c.strip("_").lower()

def standardize_columns(df):
    df = df.copy()
    df.columns = [clean_col(c) for c in df.columns]
    aliases = {
        "date": "date",
        "sales_date": "date",
        "sku": "sku_id",
        "sku_code": "sku_id",
        "product_sku": "sku_id",
        "units_sold": "units_sold",
        "units": "units_sold",
        "quantity": "units_sold",
        "revenue": "revenue",
        "sales": "revenue",
        "price": "unit_price",
        "selling_price": "unit_price",
        "promotion": "promotion",
        "promo": "promotion",
        "snapshot_date": "snapshot_date",
        "current_stock": "current_stock",
        "on_hand": "current_stock",
        "on_order": "on_order",
        "lead_time_days": "lead_time_days",
        "safety_stock": "safety_stock",
        "reorder_point": "reorder_point",
        "inventory_value": "inventory_value",
        "product_name": "product_name",
        "category": "category",
        "subcategory": "subcategory",
        "launch_date": "launch_date",
        "cost_price": "cost_price",
        "gross_margin_per_unit": "gross_margin_per_unit",
        "year": "year",
        "month": "month",
        "quarter": "quarter",
        "week": "week",
        "day_of_week": "day_of_week",
        "is_weekend": "is_weekend",
        "season": "season",
        "holiday": "holiday",
        "is_holiday": "is_holiday",
        "promotion_event": "promotion_event",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    return df

def to_bool01(s):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).clip(0,1).astype(int)
    return s.astype(str).str.strip().str.lower().isin(
        ["1","true","yes","y","promo","promotion","holiday"]
    ).astype(int)

def wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom else np.nan

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
