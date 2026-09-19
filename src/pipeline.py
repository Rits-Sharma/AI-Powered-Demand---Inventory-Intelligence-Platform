import pandas as pd
import numpy as np
from src.config import *
from src.utils import standardize_columns, to_bool01

def read_csv_flexible(path, alt=None):
    if path.exists():
        return pd.read_csv(path)
    if alt is not None and alt.exists():
        return pd.read_csv(alt)
    raise FileNotFoundError(
        f"Missing required file: {path.name}. Put it in {RAW_DIR}."
    )

def load_raw():
    sales = read_csv_flexible(SALES_FILE)
    sku = read_csv_flexible(SKU_FILE)
    calendar = read_csv_flexible(CALENDAR_FILE, CALENDAR_ALT_FILE)
    inventory = read_csv_flexible(INVENTORY_FILE)
    return sales, sku, calendar, inventory

def clean_sales(df):
    df = standardize_columns(df)
    required = ["date", "sku_id", "units_sold"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"sales_daily.csv needs column: {c}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    for c in ["revenue", "unit_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "promotion" in df.columns:
        df["promotion"] = to_bool01(df["promotion"])
    else:
        df["promotion"] = 0
    df = df.dropna(subset=["date", "sku_id", "units_sold"])
    df = df[df["units_sold"] >= 0].copy()
    if "revenue" in df.columns:
        df["revenue"] = df["revenue"].fillna(0).clip(lower=0)
    else:
        df["revenue"] = df["units_sold"] * df["unit_price"].fillna(0)
    df = df.sort_values(["sku_id", "date"])
    df = df.drop_duplicates(["date", "sku_id"], keep="last")
    return df

def clean_sku(df):
    df = standardize_columns(df)
    if "sku_id" not in df.columns:
        raise ValueError("sku_master.csv needs SKU.")
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    for c in ["launch_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["cost_price","selling_price","gross_margin_per_unit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("sku_id")

def clean_calendar(df):
    df = standardize_columns(df)
    if "date" not in df.columns:
        raise ValueError("calendar.csv needs date.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["year","month","quarter","week","day_of_week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["is_weekend","is_holiday","promotion_event"]:
        if c in df.columns:
            df[c] = to_bool01(df[c])
    df = df.dropna(subset=["date"]).drop_duplicates("date")
    return df

def clean_inventory(df):
    df = standardize_columns(df)
    required = ["snapshot_date","sku_id","current_stock"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"inventory_snapshots.csv needs column: {c}")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["sku_id"] = df["sku_id"].astype(str).str.strip()
    numeric = ["current_stock","on_order","lead_time_days","safety_stock","reorder_point","inventory_value"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["snapshot_date","sku_id","current_stock"])
    return df.sort_values(["sku_id","snapshot_date"]).drop_duplicates(["snapshot_date","sku_id"], keep="last")

def build_analysis_ready():
    sales_raw, sku_raw, cal_raw, inv_raw = load_raw()
    sales = clean_sales(sales_raw)
    sku = clean_sku(sku_raw)
    calendar = clean_calendar(cal_raw)
    inventory = clean_inventory(inv_raw)

    # Sales + calendar
    out = sales.merge(calendar, on="date", how="left", suffixes=("","_cal"))
    # Product master
    out = out.merge(sku, on="sku_id", how="left", suffixes=("","_sku"))

    # Inventory as-of join: latest known snapshot for each SKU on/before sales date.
    inv = inventory.sort_values(["sku_id","snapshot_date"]).copy()
    out = out.sort_values(["sku_id","date"])
    out = pd.merge_asof(
        out.sort_values("date"),
        inv.sort_values("snapshot_date"),
        left_on="date",
        right_on="snapshot_date",
        by="sku_id",
        direction="backward",
        suffixes=("","_inv")
    ).sort_values(["sku_id","date"])

    out["inventory_match"] = out["current_stock"].notna().astype(int)

    # Fill calendar features from date when missing.
    out["year"] = out["year"].fillna(out["date"].dt.year)
    out["month"] = out["month"].fillna(out["date"].dt.month)
    out["quarter"] = out["quarter"].fillna(out["date"].dt.quarter)
    out["week"] = out["week"].fillna(out["date"].dt.isocalendar().week.astype(int))
    out["day_of_week"] = out["day_of_week"].fillna(out["date"].dt.dayofweek)
    out["is_weekend"] = out["is_weekend"].fillna(out["day_of_week"].isin([5,6]).astype(int))

    out.to_csv(PROCESSED_DIR / "analysis_ready.csv", index=False)

    quality = {
        "sales_rows": int(len(sales)),
        "sales_skus": int(sales["sku_id"].nunique()),
        "sales_date_min": str(sales["date"].min().date()),
        "sales_date_max": str(sales["date"].max().date()),
        "sku_rows": int(len(sku)),
        "calendar_rows": int(len(calendar)),
        "inventory_rows": int(len(inventory)),
        "inventory_unique_skus": int(inventory["sku_id"].nunique()),
        "sales_skus_missing_in_master": int((~sales["sku_id"].isin(sku["sku_id"])).sum()),
        "sales_rows_with_inventory": int(out["inventory_match"].sum()),
        "missing_values_top": out.isna().sum().sort_values(ascending=False).head(15).to_dict()
    }
    import json
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
    print("Saved:", PROCESSED_DIR / "analysis_ready.csv")
    print(json.dumps(quality, indent=2, default=str))
    return out

if __name__ == "__main__":
    build_analysis_ready()
