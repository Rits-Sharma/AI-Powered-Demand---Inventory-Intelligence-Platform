import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import *
from src.features import FEATURES
from src.utils import wape

def main():
    path = PROCESSED_DIR / "model_features.csv"
    if not path.exists():
        from src.features import main as build_features
        build_features()

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(weeks=8)

    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()

    X_train = train[FEATURES].astype(float)
    y_train = train["units_sold"].astype(float)
    X_test = test[FEATURES].astype(float)
    y_test = test["units_sold"].astype(float)

    model = XGBRegressor(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, None)
    baseline = np.clip(test["lag_7"].values, 0, None)

    metrics = {
        "model": "XGBoost",
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "xgboost_wape": wape(y_test, pred),
        "baseline_wape": wape(y_test, baseline),
        "xgboost_mae": float(mean_absolute_error(y_test, pred)),
        "xgboost_rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "improvement_vs_baseline_pct": float(
            (wape(y_test, baseline) - wape(y_test, pred)) / wape(y_test, baseline) * 100
        ) if wape(y_test, baseline) else None
    }

    test_out = test[["date","sku_id","units_sold"]].copy()
    test_out["baseline_pred"] = baseline
    test_out["xgboost_pred"] = pred
    test_out.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)

    joblib.dump({"model": model, "features": FEATURES}, MODELS_DIR / "xgboost_demand_model.joblib")
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(REPORTS_DIR / "xgboost_feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    return model, test_out

if __name__ == "__main__":
    main()
