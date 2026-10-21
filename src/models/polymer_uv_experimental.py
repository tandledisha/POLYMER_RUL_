
"""Experimental UV degradation prediction and data comparison.
Not a validated RUL model.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT / "Data" / "processed" / "polymer_ml"
    / "best_uv_mass_GradientBoosting.joblib"
)
DATA_PATH = (
    PROJECT_ROOT / "Data" / "processed" / "polymer_training"
    / "polymer_training_uv.csv"
)

SUPPORTED_UV_MATERIALS = {
    "HDPE": "HDPE1",
    "LDPE": "LDPE",
    "PP": "PP1",
}


def predict_uv_degradation(material: str, exposure_time_hours: float) -> dict:
    """Return an experimental estimate and the matching observed trajectory."""
    material_key = str(material).strip().upper()

    if material_key not in SUPPORTED_UV_MATERIALS:
        raise ValueError("Choose HDPE, LDPE, or PP.")

    hours = float(exposure_time_hours)
    if not np.isfinite(hours) or not 0 <= hours <= 800:
        raise ValueError("Exposure time must be between 0 and 800 hours.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"UV model not found: {MODEL_PATH}")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"UV training data not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    sample_material = SUPPORTED_UV_MATERIALS[material_key]

    def make_features(times):
        frame = pd.DataFrame({
            "exposure_time_hours": times,
            "time_squared": np.asarray(times, dtype=float) ** 2,
            "material": sample_material,
            "exposure_condition": "UV",
        })
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            frame = frame.loc[:, list(expected)]
        return frame

    raw_prediction = float(
        np.asarray(model.predict(make_features([hours]))).reshape(-1)[0]
    )
    if not np.isfinite(raw_prediction):
        raise ValueError("The UV model returned a non-finite prediction.")

    times = np.arange(0, 801, 25, dtype=float)
    curve = np.asarray(model.predict(make_features(times)), dtype=float)
    if not np.isfinite(curve).all():
        raise ValueError("The UV model returned non-finite curve values.")

    data = pd.read_csv(DATA_PATH)
    observed = data.loc[
        (data["material"] == sample_material)
        & (data["exposure_condition"] == "UV"),
        ["exposure_time_hours", "target_degradation"],
    ].sort_values("exposure_time_hours")

    if observed.empty:
        raise ValueError(f"No observations found for {sample_material}.")

    return {
        "material": material_key,
        "sample_trajectory": sample_material,
        "exposure_time_hours": hours,
        "raw_degradation_estimate": raw_prediction,
        "prediction_curve": pd.DataFrame({
            "Exposure time (hours)": times,
            "Model estimate (experimental)": curve,
        }),
        "observed_data": observed.rename(columns={
            "exposure_time_hours": "Exposure time (hours)",
            "target_degradation": "Observed target degradation",
        }),
        "prediction_type": "experimental_uv_degradation",
        "validated_rul": False,
        "warning": (
            "Experimental model output and a single sample trajectory only. "
            "This is not validated physical degradation or RUL. "
            "The target is normalized degradation, not a validated failure criterion."
        ),
    }
