from src.pipeline import build_analysis_ready
from src.eda import main as run_eda
from src.features import main as build_features
from src.baseline import main as run_baseline
from src.forecast_model import main as run_model
from src.risk_scoring import main as run_risk

print("\n=== FORESIGHT PIPELINE START ===")
build_analysis_ready()
run_eda()
build_features()
run_baseline()
run_model()
run_risk()
print("\n=== FORESIGHT PIPELINE COMPLETE ===")
print("Next: streamlit run app/dashboard.py")