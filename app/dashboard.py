import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="FORESIGHT", layout="wide")
st.title("Project FORESIGHT")
st.caption("AI-powered demand forecasting and inventory intelligence")

ready_path = PROC / "analysis_ready.csv"
risk_path = PROC / "risk_scores.csv"
pred_path = PROC / "test_predictions.csv"
metrics_path = REPORTS / "model_metrics.json"

if not ready_path.exists():
    st.error("Processed data not found. Run `python run_pipeline.py` first.")
    st.stop()

df = pd.read_csv(ready_path, parse_dates=["date"])
risk = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
pred = pd.read_csv(pred_path, parse_dates=["date"]) if pred_path.exists() else pd.DataFrame()

# Sidebar filters
st.sidebar.header("Filters")
categories = sorted(df["category"].dropna().astype(str).unique()) if "category" in df.columns else []
selected_cat = st.sidebar.multiselect("Category", categories, default=categories)
if selected_cat and "category" in df.columns:
    view = df[df["category"].astype(str).isin(selected_cat)].copy()
else:
    view = df.copy()

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Revenue", f"₹{view['revenue'].sum():,.0f}")
c2.metric("Units Sold", f"{view['units_sold'].sum():,.0f}")
c3.metric("SKUs", f"{view['sku_id'].nunique():,}")
c4.metric("Avg Daily Units", f"{view.groupby('date')['units_sold'].sum().mean():,.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview","Sales","Inventory Risk","Forecast"])

with tab1:
    daily = view.groupby("date", as_index=False).agg(units_sold=("units_sold","sum"), revenue=("revenue","sum"))
    st.plotly_chart(px.line(daily, x="date", y="units_sold", title="Daily Demand"), use_container_width=True)
    if "category" in view.columns:
        cat = view.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(cat, x="category", y="revenue", title="Revenue by Category"), use_container_width=True)

with tab2:
    sku = view.groupby("sku_id", as_index=False).agg(
        units_sold=("units_sold","sum"), revenue=("revenue","sum")
    ).sort_values("revenue", ascending=False)
    st.subheader("Top 20 SKUs")
    st.dataframe(sku.head(20), use_container_width=True)
    st.plotly_chart(px.bar(sku.head(15), x="sku_id", y="revenue", title="Top SKU Revenue"), use_container_width=True)

with tab3:
    if risk.empty:
        st.info("Risk output not found.")
    else:
        counts = risk["risk_category"].value_counts().reset_index()
        counts.columns = ["risk_category","sku_count"]
        st.plotly_chart(px.bar(counts, x="risk_category", y="sku_count", title="Risk Categories"), use_container_width=True)
        cols = [c for c in ["sku_id","product_name","category","current_stock","avg_forecast_daily","days_of_cover","risk_category"] if c in risk.columns]
        st.dataframe(risk[cols].sort_values("days_of_cover").head(50), use_container_width=True)

with tab4:
    if pred.empty:
        st.info("Forecast output not found.")
    else:
        sku_list = sorted(pred["sku_id"].astype(str).unique())
        chosen = st.selectbox("Select SKU", sku_list)
        p = pred[pred["sku_id"].astype(str) == chosen].sort_values("date")
        chart = p.melt(id_vars=["date"], value_vars=["units_sold","xgboost_pred"], var_name="series", value_name="units")
        st.plotly_chart(px.line(chart, x="date", y="units", color="series", title=f"Actual vs XGBoost — {chosen}"), use_container_width=True)