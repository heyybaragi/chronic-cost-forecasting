import joblib
import pandas as pd

_MODEL = None
_CUTOFFS = None


def _load_artifacts():
    """Load model and tier cutoffs once, reuse across calls."""
    global _MODEL, _CUTOFFS
    if _MODEL is None:
        _MODEL = joblib.load('src/xgb_cost_model.pkl')
    if _CUTOFFS is None:
        _CUTOFFS = joblib.load('src/risk_tier_cutoffs.pkl')
    return _MODEL, _CUTOFFS


def predict_cost_and_tier(patient_features: pd.DataFrame) -> dict:
    """
    Takes a single-row DataFrame (from build_patient_features) and returns
    predicted cost plus assigned risk tier.
    """
    model, cutoffs = _load_artifacts()
    predicted_cost = float(model.predict(patient_features)[0])

    if predicted_cost <= cutoffs['low_cutoff']:
        tier = 'Low'
    elif predicted_cost <= cutoffs['high_cutoff']:
        tier = 'Medium'
    else:
        tier = 'High'

    return {
        'predicted_cost': predicted_cost,
        'risk_tier': tier
    }