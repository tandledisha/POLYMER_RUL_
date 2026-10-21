import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# POLYLIFE - ENVIRONMENTAL POLYMER PREDICTOR
# ============================================================
#
# CURRENT MODEL
# -------------
# Optimized ExtraTrees degradation model
#
# Validated target:
#     degradation_index
#
# Current optimized validation:
#     R² ≈ 0.8821 (88.21%)
#
# CURRENTLY SUPPORTED POLYMER
# ---------------------------
# Polyethylene / PE
#
# INPUTS
# ------
# polymer_type
# baseline_performance
# temperature_C
# humidity_percent
# uv_irradiance_W_m2
# aging_time_hours
#
# OPTIONAL:
# uv_dose_J_m2
#
# OUTPUTS
# -------
# current_degradation
# current_performance
# estimated_RUL_hours
# lower_RUL_hours
# upper_RUL_hours
# status
#
# IMPORTANT
# ---------
# The current model was trained on PE environmental-aging
# data.
#
# The forecast assumes the entered environmental conditions
# remain constant over the future scenario.
#
# RUL is a prototype scenario estimate based on a selected
# performance failure threshold.
#
# It is NOT yet a validated real irrigation-component
# lifetime.
#
# ============================================================


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)


MODEL_DIR = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "nist_optimized_degradation"
)


MODEL_FILE = (
    MODEL_DIR
    / "optimization_best_model.joblib"
)


METADATA_FILE = (
    MODEL_DIR
    / "optimization_metadata.json"
)


OOF_FILE = (
    MODEL_DIR
    / "optimization_oof_predictions.csv"
)


# ============================================================
# 2. CONSTANTS
# ============================================================

SUPPORTED_POLYMERS = [
    "Polyethylene",
    "PE",
]


# The current NIST model was developed using aging windows
# up to approximately 2016 hours.
VALIDATED_TIME_MIN_HOURS = 120.0
VALIDATED_TIME_MAX_HOURS = 2016.0


DEFAULT_FAILURE_THRESHOLD_PERCENT = 70.0


RUL_GRID_STEP_HOURS = 12.0


# ============================================================
# 3. CHECK MODEL
# ============================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(

        "\nPolyLife optimized model was not found:\n"
        f"{MODEL_FILE}\n\n"
        "Run first:\n"
        "py src/models/optimize_nist_degradation_model.py"

    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

MODEL = joblib.load(
    MODEL_FILE
)


# ------------------------------------------------------------
# Avoid unnecessary parallel-worker warnings during inference.
# ------------------------------------------------------------

if hasattr(
    MODEL,
    "set_params"
):

    try:

        MODEL.set_params(
            n_jobs=1
        )

    except Exception:

        pass


# ============================================================
# 5. LOAD MODEL METADATA
# ============================================================

if METADATA_FILE.exists():

    try:

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            MODEL_METADATA = json.load(
                f
            )

    except Exception:

        MODEL_METADATA = {}

else:

    MODEL_METADATA = {}


# ============================================================
# 6. LOAD OOF RESIDUALS
# ============================================================
#
# We use the out-of-fold prediction errors to obtain a
# conservative residual uncertainty estimate.
#
# This is a prototype uncertainty estimate and NOT a formally
# calibrated conformal interval.
#
# ============================================================

RESIDUAL_Q90 = None


if OOF_FILE.exists():

    try:

        oof = pd.read_csv(
            OOF_FILE
        )


        best_config = (
            MODEL_METADATA
            .get(
                "best_config",
                None
            )
        )


        if (
            best_config is not None
            and
            "config" in oof.columns
        ):

            oof = oof[
                oof[
                    "config"
                ]
                ==
                best_config
            ].copy()


        if (
            "actual_degradation"
            in
            oof.columns
            and
            "predicted_degradation"
            in
            oof.columns
        ):

            residuals = (

                oof[
                    "actual_degradation"
                ]

                -

                oof[
                    "predicted_degradation"
                ]

            ).abs()


            residuals = residuals[
                np.isfinite(
                    residuals
                )
            ]


            if len(
                residuals
            ) > 0:

                RESIDUAL_Q90 = float(

                    np.quantile(
                        residuals,
                        0.90
                    )

                )

    except Exception:

        RESIDUAL_Q90 = None


if (
    RESIDUAL_Q90 is None
    or
    not np.isfinite(
        RESIDUAL_Q90
    )
):

    RESIDUAL_Q90 = 0.10


# ============================================================
# 7. HELPER - VALIDATE POLYMER
# ============================================================

def validate_polymer(
    polymer_type
):

    polymer_clean = (
        str(
            polymer_type
        )
        .strip()
    )


    if polymer_clean not in SUPPORTED_POLYMERS:

        raise ValueError(

            "The current trained environmental model "
            "supports only Polyethylene (PE)."

        )


    return "Polyethylene"


# ============================================================
# 8. FEATURE BUILDER
# ============================================================

def build_features(

    polymer_type,

    baseline_performance,

    temperature_C,

    humidity_percent,

    uv_irradiance_W_m2,

    aging_time_hours,

    uv_dose_J_m2=None,

):
    """
    Build the exact feature representation expected by the
    optimized ExtraTrees degradation model.

    Environmental conditions are treated as constant over the
    scenario duration.
    """

    polymer_clean = validate_polymer(
        polymer_type
    )


    baseline_performance = float(
        baseline_performance
    )

    temperature_C = float(
        temperature_C
    )

    humidity_percent = float(
        humidity_percent
    )

    uv_irradiance_W_m2 = float(
        uv_irradiance_W_m2
    )

    aging_time_hours = float(
        aging_time_hours
    )


    if uv_dose_J_m2 is not None:

        uv_dose_J_m2 = float(
            uv_dose_J_m2
        )


    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if baseline_performance <= 0:

        raise ValueError(
            "Baseline performance must be greater than zero."
        )


    if not (
        0.0
        <= humidity_percent
        <= 100.0
    ):

        raise ValueError(
            "Humidity must be between 0 and 100%."
        )


    if temperature_C < -100:

        raise ValueError(
            "Temperature is outside a reasonable range."
        )


    if uv_irradiance_W_m2 < 0:

        raise ValueError(
            "UV irradiance cannot be negative."
        )


    if aging_time_hours < 0:

        raise ValueError(
            "Aging time cannot be negative."
        )


    if (
        uv_dose_J_m2 is not None
        and
        uv_dose_J_m2 < 0
    ):

        raise ValueError(
            "UV dose cannot be negative."
        )


    # ========================================================
    # EXPOSURE
    # ========================================================

    window_duration_hours = max(
        aging_time_hours,
        1.0
    )


    # Constant-condition UV dose.
    # W/m² × seconds = J/m²
    calculated_uv_dose = (

        uv_irradiance_W_m2

        *

        aging_time_hours

        *

        3600.0

    )


    if uv_dose_J_m2 is not None:

        effective_uv_dose = max(
            0.0,
            uv_dose_J_m2
        )

    else:

        effective_uv_dose = max(
            0.0,
            calculated_uv_dose
        )


    # ========================================================
    # TEMPERATURE FEATURES
    # ========================================================

    temperature_min_C = (
        temperature_C
    )

    temperature_max_C = (
        temperature_C
    )

    temperature_range_C = 0.0

    temperature_squared = (
        temperature_C
        ** 2
    )


    thermal_exposure = (

        temperature_C

        *

        window_duration_hours

    )


    thermal_exposure_per_hour = (

        thermal_exposure

        /

        window_duration_hours

    )


    # ========================================================
    # HUMIDITY FEATURES
    # ========================================================

    humidity_min_percent = (
        humidity_percent
    )

    humidity_max_percent = (
        humidity_percent
    )

    humidity_range_percent = 0.0

    humidity_squared = (
        humidity_percent
        ** 2
    )


    humidity_exposure = (

        humidity_percent

        *

        window_duration_hours

    )


    humidity_exposure_per_hour = (

        humidity_exposure

        /

        window_duration_hours

    )


    # ========================================================
    # UV FEATURES
    # ========================================================

    log_uv_dose = np.log1p(
        effective_uv_dose
    )


    log_uv_irradiance = np.log1p(
        max(
            uv_irradiance_W_m2,
            0.0
        )
    )


    uv_dose_per_hour = (

        effective_uv_dose

        /

        window_duration_hours

    )


    # ========================================================
    # TIME FEATURES
    # ========================================================

    log_aging_time = np.log1p(
        max(
            aging_time_hours,
            0.0
        )
    )


    aging_time_squared = (
        aging_time_hours
        ** 2
    )


    # ========================================================
    # INTERACTION FEATURES
    # ========================================================

    temperature_x_uv = (

        temperature_C

        *

        log_uv_dose

    )


    humidity_x_uv = (

        humidity_percent

        *

        log_uv_dose

    )


    temperature_x_humidity = (

        temperature_C

        *

        humidity_percent

    )


    temperature_x_uv_dose = (

        temperature_C

        *

        np.log1p(
            max(
                effective_uv_dose,
                0.0
            )
        )

    )


    humidity_x_uv_dose = (

        humidity_percent

        *

        np.log1p(
            max(
                effective_uv_dose,
                0.0
            )
        )

    )


    # ========================================================
    # CUMULATIVE FEATURES
    # ========================================================
    #
    # Constant-condition scenario:
    #
    # cumulative exposure =
    #       current exposure intensity × current time
    #
    # ========================================================

    cumulative_uv_dose = (
        effective_uv_dose
    )


    log_cumulative_uv = np.log1p(
        max(
            cumulative_uv_dose,
            0.0
        )
    )


    cumulative_thermal_exposure = (
        thermal_exposure
    )


    cumulative_humidity_exposure = (
        humidity_exposure
    )


    cumulative_uv_per_hour = (

        cumulative_uv_dose

        /

        max(
            aging_time_hours,
            1.0
        )

    )


    cumulative_thermal_per_hour = (

        cumulative_thermal_exposure

        /

        max(
            aging_time_hours,
            1.0
        )

    )


    cumulative_humidity_per_hour = (

        cumulative_humidity_exposure

        /

        max(
            aging_time_hours,
            1.0
        )

    )


    # ========================================================
    # EXACT TRAINING FEATURE ORDER
    # ========================================================

    features = {

        "baseline_performance":
            baseline_performance,

        "aging_time_hours":
            aging_time_hours,

        "log_aging_time":
            log_aging_time,

        "aging_time_squared":
            aging_time_squared,

        "temperature_C":
            temperature_C,

        "temperature_min_C":
            temperature_min_C,

        "temperature_max_C":
            temperature_max_C,

        "temperature_range_C":
            temperature_range_C,

        "temperature_squared":
            temperature_squared,

        "humidity_percent":
            humidity_percent,

        "humidity_min_percent":
            humidity_min_percent,

        "humidity_max_percent":
            humidity_max_percent,

        "humidity_range_percent":
            humidity_range_percent,

        "humidity_squared":
            humidity_squared,

        "uv_irradiance_mean_W_m2":
            uv_irradiance_W_m2,

        "uv_irradiance_max_W_m2":
            uv_irradiance_W_m2,

        "log_uv_irradiance":
            log_uv_irradiance,

        "uv_dose_J_m2":
            effective_uv_dose,

        "log_uv_dose":
            log_uv_dose,

        "uv_dose_per_hour":
            uv_dose_per_hour,

        "thermal_exposure":
            thermal_exposure,

        "thermal_exposure_per_hour":
            thermal_exposure_per_hour,

        "humidity_exposure":
            humidity_exposure,

        "humidity_exposure_per_hour":
            humidity_exposure_per_hour,

        "cumulative_uv_dose":
            cumulative_uv_dose,

        "log_cumulative_uv":
            log_cumulative_uv,

        "cumulative_uv_per_hour":
            cumulative_uv_per_hour,

        "cumulative_thermal_exposure":
            cumulative_thermal_exposure,

        "cumulative_thermal_per_hour":
            cumulative_thermal_per_hour,

        "cumulative_humidity_exposure":
            cumulative_humidity_exposure,

        "cumulative_humidity_per_hour":
            cumulative_humidity_per_hour,

        "temperature_x_uv":
            temperature_x_uv,

        "humidity_x_uv":
            humidity_x_uv,

        "temperature_x_humidity":
            temperature_x_humidity,

        "temperature_x_uv_dose":
            temperature_x_uv_dose,

        "humidity_x_uv_dose":
            humidity_x_uv_dose,

    }


    # `polymer_clean` is currently only validated because the
    # trained PE model does not have polymer category features.
    # Do not silently pretend another polymer is supported.

    _ = polymer_clean


    return pd.DataFrame(
        [features]
    )


# ============================================================
# 9. PREDICT CURRENT DEGRADATION
# ============================================================

def predict_degradation(

    polymer_type,

    baseline_performance,

    temperature_C,

    humidity_percent,

    uv_irradiance_W_m2,

    aging_time_hours,

    uv_dose_J_m2=None,

):
    """
    Predict current normalized degradation and current polymer
    performance.
    """

    X = build_features(

        polymer_type=
            polymer_type,

        baseline_performance=
            baseline_performance,

        temperature_C=
            temperature_C,

        humidity_percent=
            humidity_percent,

        uv_irradiance_W_m2=
            uv_irradiance_W_m2,

        aging_time_hours=
            aging_time_hours,

        uv_dose_J_m2=
            uv_dose_J_m2,

    )


    predicted_degradation = (
        MODEL.predict(
            X
        )[0]
    )


    predicted_degradation = float(
        np.clip(
            predicted_degradation,
            0.0,
            1.0
        )
    )


    predicted_performance = (

        float(
            baseline_performance
        )

        *

        (
            1.0
            -
            predicted_degradation
        )

    )


    predicted_performance = max(
        0.0,
        float(
            predicted_performance
        )
    )


    return {

        "predicted_degradation":
            predicted_degradation,

        "predicted_performance":
            predicted_performance,

        "uncertainty_degradation_q90":
            RESIDUAL_Q90,

    }


# ============================================================
# 10. INTERPOLATED THRESHOLD CROSSING
# ============================================================

def find_threshold_crossing(
    times,
    degradation,
    threshold,
):
    """
    Find the first time degradation reaches the specified
    threshold.

    Linear interpolation is used between grid points.
    """

    for index in range(
        1,
        len(times)
    ):

        previous_degradation = (
            degradation[
                index - 1
            ]
        )

        current_degradation = (
            degradation[
                index
            ]
        )


        if (

            previous_degradation
            <
            threshold

            and

            current_degradation
            >=
            threshold

        ):

            previous_time = (
                times[
                    index - 1
                ]
            )

            current_time = (
                times[
                    index
                ]
            )


            denominator = (

                current_degradation
                -
                previous_degradation

            )


            if abs(
                denominator
            ) < 1e-12:

                return float(
                    current_time
                )


            fraction = (

                threshold
                -
                previous_degradation

            ) / denominator


            crossing_time = (

                previous_time

                +

                fraction
                *
                (
                    current_time
                    -
                    previous_time
                )

            )


            return float(
                crossing_time
            )


    return None


# ============================================================
# 11. PREDICT RUL
# ============================================================

def predict_rul(

    polymer_type,

    baseline_performance,

    temperature_C,

    humidity_percent,

    uv_irradiance_W_m2,

    aging_time_hours,

    performance_failure_threshold_percent=
        DEFAULT_FAILURE_THRESHOLD_PERCENT,

    uv_dose_J_m2=None,

):
    """
    Estimate prototype RUL from the environmental degradation
    model.

    Failure threshold is expressed as remaining polymer
    performance percentage of baseline.

    Example:

        threshold = 70%

    means:

        failure degradation threshold = 1 - 0.70 = 0.30
    """

    # --------------------------------------------------------
    # Validate polymer
    # --------------------------------------------------------

    polymer_clean = validate_polymer(
        polymer_type
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold_percent = float(
        performance_failure_threshold_percent
    )


    if not (
        0.0
        <
        threshold_percent
        <
        100.0
    ):

        raise ValueError(
            "Failure threshold must be between 0 and 100%."
        )


    degradation_threshold = (

        1.0

        -

        (
            threshold_percent
            /
            100.0
        )

    )


    # --------------------------------------------------------
    # Current prediction
    # --------------------------------------------------------

    current_result = predict_degradation(

        polymer_type=
            polymer_clean,

        baseline_performance=
            baseline_performance,

        temperature_C=
            temperature_C,

        humidity_percent=
            humidity_percent,

        uv_irradiance_W_m2=
            uv_irradiance_W_m2,

        aging_time_hours=
            aging_time_hours,

        uv_dose_J_m2=
            uv_dose_J_m2,

    )


    current_degradation = (
        current_result[
            "predicted_degradation"
        ]
    )


    current_performance = (
        current_result[
            "predicted_performance"
        ]
    )


    # ========================================================
    # CASE 2 - CURRENT AGE ALREADY OUTSIDE TRAINING RANGE
    # ========================================================

    if (
        aging_time_hours
        >
        VALIDATED_TIME_MAX_HOURS
    ):

        return {

            "polymer_type":
                polymer_clean,

            "baseline_performance":
                float(
                    baseline_performance
                ),

            "temperature_C":
                float(
                    temperature_C
                ),

            "humidity_percent":
                float(
                    humidity_percent
                ),

            "uv_irradiance_W_m2":
                float(
                    uv_irradiance_W_m2
                ),

            "aging_time_hours":
                float(
                    aging_time_hours
                ),

            "failure_threshold_percent":
                threshold_percent,

            "degradation_threshold":
                degradation_threshold,

            "current_degradation":
                current_degradation,

            "current_performance":
                current_performance,

            "estimated_RUL_hours":
                0.0,

            "lower_RUL_hours":
                0.0,

            "upper_RUL_hours":
                0.0,

            "uncertainty_degradation_q90":
                RESIDUAL_Q90,

            "status":
                "outside_validated_age_range",

            "forecast_limit_hours":
                VALIDATED_TIME_MAX_HOURS,

            "within_validated_forecast":
                False,

            "uncertainty_method":
                (
                    "90th-percentile out-of-fold "
                    "residual prototype"
                ),

            "scenario_assumption":
                (
                    "Temperature, humidity and UV "
                    "conditions remain constant."
                ),

        }


    # ========================================================
    # CASE 1 - ALREADY AT / BEYOND THRESHOLD
    # ========================================================

    if (
        current_degradation
        >=
        degradation_threshold
    ):

        return {

            "polymer_type":
                polymer_clean,

            "baseline_performance":
                float(
                    baseline_performance
                ),

            "temperature_C":
                float(
                    temperature_C
                ),

            "humidity_percent":
                float(
                    humidity_percent
                ),

            "uv_irradiance_W_m2":
                float(
                    uv_irradiance_W_m2
                ),

            "aging_time_hours":
                float(
                    aging_time_hours
                ),

            "failure_threshold_percent":
                threshold_percent,

            "degradation_threshold":
                degradation_threshold,

            "current_degradation":
                current_degradation,

            "current_performance":
                current_performance,

            "estimated_RUL_hours":
                0.0,

            "lower_RUL_hours":
                0.0,

            "upper_RUL_hours":
                0.0,

            "uncertainty_degradation_q90":
                RESIDUAL_Q90,

            "status":
                "threshold_reached",

            "forecast_limit_hours":
                VALIDATED_TIME_MAX_HOURS,

            "within_validated_forecast":
                True,

            "uncertainty_method":
                (
                    "90th-percentile out-of-fold "
                    "residual prototype"
                ),

            "scenario_assumption":
                (
                    "Temperature, humidity and UV "
                    "conditions remain constant."
                ),

        }


    # ========================================================
    # CASE 3 - BUILD FUTURE SCENARIO
    # ========================================================

    start_time = max(
        float(
            aging_time_hours
        ),
        0.0
    )


    times = np.arange(

        start_time,

        VALIDATED_TIME_MAX_HOURS
        +
        RUL_GRID_STEP_HOURS,

        RUL_GRID_STEP_HOURS,

    )


    # Ensure final horizon included.
    if (
        len(times) == 0
        or
        times[-1]
        <
        VALIDATED_TIME_MAX_HOURS
    ):

        times = np.append(
            times,
            VALIDATED_TIME_MAX_HOURS
        )


    # --------------------------------------------------------
    # Future degradation
    # --------------------------------------------------------

    # Build the exact same feature rows, then predict them together.
    # Keeping build_features() preserves existing validation and formulas.
    future_frames = [
        build_features(
            polymer_type=polymer_clean,
            baseline_performance=baseline_performance,
            temperature_C=temperature_C,
            humidity_percent=humidity_percent,
            uv_irradiance_W_m2=uv_irradiance_W_m2,
            aging_time_hours=float(future_time),
            uv_dose_J_m2=None,
        )
        for future_time in times
    ]

    future_X = pd.concat(future_frames, ignore_index=True)

    # Preserve the model's expected feature order.
    if hasattr(MODEL, "feature_names_in_"):
        future_X = future_X.loc[:, list(MODEL.feature_names_in_)]

    degradation_values = np.clip(
        MODEL.predict(future_X),
        0.0,
        1.0,
    ).astype(float)


    # ========================================================
    # CENTRAL CROSSING
    # ========================================================

    central_crossing = (
        find_threshold_crossing(

            times=
                times,

            degradation=
                degradation_values,

            threshold=
                degradation_threshold,

        )
    )


    # ========================================================
    # UNCERTAINTY TRAJECTORIES
    # ========================================================

    q90 = float(
        RESIDUAL_Q90
    )


    # Higher degradation -> earlier failure
    high_degradation = np.clip(

        degradation_values
        +
        q90,

        0.0,
        1.0

    )


    # Lower degradation -> later failure
    low_degradation = np.clip(

        degradation_values
        -
        q90,

        0.0,
        1.0

    )


    early_crossing = (
        find_threshold_crossing(

            times=
                times,

            degradation=
                high_degradation,

            threshold=
                degradation_threshold,

        )
    )


    late_crossing = (
        find_threshold_crossing(

            times=
                times,

            degradation=
                low_degradation,

            threshold=
                degradation_threshold,

        )
    )


    # ========================================================
    # CONVERT CROSSING TIMES TO RUL
    # ========================================================

    if central_crossing is None:

        estimated_rul = None

    else:

        estimated_rul = max(

            0.0,

            central_crossing
            -
            start_time

        )


    if early_crossing is None:

        lower_rul = None

    else:

        lower_rul = max(

            0.0,

            early_crossing
            -
            start_time

        )


    if late_crossing is None:

        upper_rul = None

    else:

        upper_rul = max(

            0.0,

            late_crossing
            -
            start_time

        )


    # ========================================================
    # CASE 4 - THRESHOLD NOT REACHED IN VALIDATED HORIZON
    # ========================================================

    if estimated_rul is None:

        remaining_validated_horizon = max(

            0.0,

            VALIDATED_TIME_MAX_HOURS
            -
            start_time

        )


        # When no threshold crossing is found, the honest
        # result is a lower-bound statement:
        #
        #     RUL > remaining validated horizon
        #
        # instead of inventing a lifetime.
        #

        return {

            "polymer_type":
                polymer_clean,

            "baseline_performance":
                float(
                    baseline_performance
                ),

            "temperature_C":
                float(
                    temperature_C
                ),

            "humidity_percent":
                float(
                    humidity_percent
                ),

            "uv_irradiance_W_m2":
                float(
                    uv_irradiance_W_m2
                ),

            "aging_time_hours":
                float(
                    aging_time_hours
                ),

            "failure_threshold_percent":
                threshold_percent,

            "degradation_threshold":
                degradation_threshold,

            "current_degradation":
                current_degradation,

            "current_performance":
                current_performance,

            "estimated_RUL_hours":
                remaining_validated_horizon,

            "lower_RUL_hours":
                (
                    max(
                        0.0,
                        lower_rul
                    )
                    if lower_rul is not None
                    else remaining_validated_horizon
                ),

            "upper_RUL_hours":
                None,

            "RUL_interpretation":
                (
                    "RUL is greater than the remaining "
                    "validated forecast horizon."
                ),

            "remaining_validated_horizon_hours":
                remaining_validated_horizon,

            "status":
                "beyond_validated_forecast",

            "forecast_limit_hours":
                VALIDATED_TIME_MAX_HOURS,

            "within_validated_forecast":
                False,

            "uncertainty_degradation_q90":
                q90,

            "uncertainty_method":
                (
                    "90th-percentile out-of-fold "
                    "residual prototype"
                ),

            "scenario_assumption":
                (
                    "Temperature, humidity and UV "
                    "conditions remain constant."
                ),

        }


    # ========================================================
    # NORMAL STATUS
    # ========================================================

    status = "normal"


    if estimated_rul <= 168:

        status = "critical"

    elif estimated_rul <= 336:

        status = "high_attention"

    elif estimated_rul <= 720:

        status = "moderate"


    # ========================================================
    # RETURN NORMAL RUL RESULT
    # ========================================================

    return {

        "polymer_type":
            polymer_clean,

        "baseline_performance":
            float(
                baseline_performance
            ),

        "temperature_C":
            float(
                temperature_C
            ),

        "humidity_percent":
            float(
                humidity_percent
            ),

        "uv_irradiance_W_m2":
            float(
                uv_irradiance_W_m2
            ),

        "aging_time_hours":
            float(
                aging_time_hours
            ),

        "failure_threshold_percent":
            threshold_percent,

        "degradation_threshold":
            degradation_threshold,

        "current_degradation":
            current_degradation,

        "current_performance":
            current_performance,

        "estimated_RUL_hours":
            estimated_rul,

        "lower_RUL_hours":
            (
                lower_rul
                if lower_rul is not None
                else 0.0
            ),

        "upper_RUL_hours":
            (
                upper_rul
                if upper_rul is not None
                else (
                    estimated_rul
                )
            ),

        "RUL_interpretation":
            "Threshold crossing predicted within validated horizon.",

        "remaining_validated_horizon_hours":
            max(
                0.0,
                VALIDATED_TIME_MAX_HOURS
                -
                start_time
            ),

        "status":
            status,

        "forecast_limit_hours":
            VALIDATED_TIME_MAX_HOURS,

        "within_validated_forecast":
            True,

        "uncertainty_degradation_q90":
            q90,

        "uncertainty_method":
            (
                "90th-percentile out-of-fold "
                "residual prototype"
            ),

        "scenario_assumption":
            (
                "Temperature, humidity and UV "
                "conditions remain constant."
            ),

    }


# ============================================================
# 12. SAVE RESULT
# ============================================================

def save_prediction(
    result,
    output_file=None,
):
    """
    Save prediction dictionary as JSON.
    """

    if output_file is None:

        output_file = (

            PROJECT_ROOT
            / "Data"
            / "processed"
            / "nist_optimized_degradation"
            / "latest_prediction.json"

        )


    output_file = Path(
        output_file
    )


    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )


    return output_file


# ============================================================
# 13. CLI TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 80
    )

    print(
        "POLYLIFE - ENVIRONMENTAL PREDICTOR TEST"
    )

    print(
        "=" * 80
    )


    print(
        "\nModel:",
        "Optimized ExtraTrees"
    )


    print(
        "Validated degradation R²:",
        MODEL_METADATA.get(
            "best_R2_percent",
            "Not available"
        )
    )


    print(
        "Residual Q90:",
        f"{RESIDUAL_Q90:.6f}"
    )


    print(
        "Supported polymer:",
        "Polyethylene (PE)"
    )


    # --------------------------------------------------------
    # DEMO TEST CASE
    # --------------------------------------------------------

    result = predict_rul(

        polymer_type=
            "Polyethylene",

        baseline_performance=
            800.0,

        temperature_C=
            27.0,

        humidity_percent=
            77.0,

        uv_irradiance_W_m2=
            2.5,

        aging_time_hours=
            500.0,

        performance_failure_threshold_percent=
            70.0,

    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEST RESULT"
    )

    print(
        "=" * 80
    )


    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


    output_file = save_prediction(
        result
    )


    print(
        "\nSaved test result:"
    )


    print(
        output_file
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "POLYLIFE ENVIRONMENTAL PREDICTOR TEST COMPLETE"
    )

    print(
        "=" * 80
    )