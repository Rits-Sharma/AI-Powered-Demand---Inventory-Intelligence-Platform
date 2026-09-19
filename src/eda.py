import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *
from src.utils import save_json

def main():
    path = PROCESSED_DIR / "analysis_ready.csv"
    if not path.exists():
        from src.pipeline import build_analysis_ready
        df = build_analysis_ready()
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    sns.set_theme(style="whitegrid")
    insights = {}

    # Daily trend
    daily = df.groupby("date", as_index=False).agg(
        units_sold=("units_sold","sum"),
        revenue=("revenue","sum")
    )
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(daily["date"], daily["units_sold"])
    ax.set_title("FORESIGHT — Daily Demand Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_daily_demand_trend.png", dpi=150)
    plt.close(fig)

    # Monthly
    df["month_period"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month_period", as_index=False).agg(
        units_sold=("units_sold","sum"), revenue=("revenue","sum")
    )
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(monthly["month_period"], monthly["revenue"], marker="o")
    ax.set_title("Monthly Revenue Trend"); ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "02_monthly_revenue.png", dpi=150); plt.close(fig)

    # Category revenue
    if "category" in df.columns:
        cat = df.groupby("category", dropna=False)["revenue"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10,5))
        cat.plot(kind="bar", ax=ax)
        ax.set_title("Revenue by Category"); ax.set_xlabel("Category"); ax.set_ylabel("Revenue")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "03_category_revenue.png", dpi=150); plt.close(fig)
        insights["top_category_by_revenue"] = str(cat.index[0]) if len(cat) else None
        insights["bottom_category_by_revenue"] = str(cat.index[-1]) if len(cat) else None

    # Season
    if "season" in df.columns:
        season = df.groupby("season")["units_sold"].mean().sort_values(ascending=False)
        insights["highest_average_demand_season"] = str(season.index[0]) if len(season) else None
        insights["lowest_average_demand_season"] = str(season.index[-1]) if len(season) else None

    # Promotion uplift
    if "promotion" in df.columns:
        promo = df.groupby("promotion")["units_sold"].mean()
        if 1 in promo.index and 0 in promo.index and promo[0] != 0:
            uplift = (promo[1] / promo[0] - 1) * 100
            insights["promotion_uplift_pct"] = round(float(uplift), 2)

        fig, ax = plt.subplots(figsize=(7,5))
        promo.rename(index={0:"No Promotion",1:"Promotion"}).plot(kind="bar", ax=ax)
        ax.set_title("Average Daily Units: Promotion vs No Promotion")
        ax.set_ylabel("Average Units Sold")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "04_promotion_uplift.png", dpi=150); plt.close(fig)

    # Top SKUs
    sku_perf = df.groupby("sku_id").agg(
        units_sold=("units_sold","sum"), revenue=("revenue","sum")
    ).sort_values("revenue", ascending=False)
    sku_perf.head(10).to_csv(REPORTS_DIR / "top_10_skus.csv")

    insights["top_10_skus_by_revenue"] = sku_perf.head(10)["revenue"].round(2).to_dict()
    insights["bottom_10_skus_by_revenue"] = sku_perf.tail(10)["revenue"].round(2).to_dict()

    # Demand volatility
    volatility = df.groupby("sku_id")["units_sold"].agg(["mean","std"]).fillna(0)
    volatility["cv"] = np.where(volatility["mean"] != 0, volatility["std"]/volatility["mean"], np.nan)
    insights["most_volatile_skus"] = volatility.sort_values("cv", ascending=False).head(10)["cv"].round(3).to_dict()

    save_json(insights, REPORTS_DIR / "eda_insights.json")
    print("EDA complete. Charts:", FIGURES_DIR)
    print("Insights:", REPORTS_DIR / "eda_insights.json")

if __name__ == "__main__":
    main()
