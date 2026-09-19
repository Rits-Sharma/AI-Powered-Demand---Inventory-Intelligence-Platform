import pandas as pd
from src.config import *
from src.utils import wape

def main():
    path = PROCESSED_DIR / "model_features.csv"
    if not path.exists():
        from src.features import main as build_features
        build_features()
    df = pd.read_csv(path, parse_dates=["date"]).sort_values(["date","sku_id"])

    cutoff = df["date"].max() - pd.Timedelta(weeks=8)
    test = df[df["date"] > cutoff].copy()
    test["baseline_pred"] = test["lag_7"].clip(lower=0)

    metrics = {
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "baseline_wape": wape(test["units_sold"], test["baseline_pred"])
    }
    save = REPORTS_DIR / "baseline_metrics.json"
    import json
    save.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return test

if __name__ == "__main__":
    main()
