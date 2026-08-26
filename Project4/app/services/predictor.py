import joblib
import pandas as pd

from pathlib import Path


# ==========================================
# PROJECT ROOT
# ==========================================

BASE_DIR = Path(
    __file__
).resolve().parents[2]


# ==========================================
# MODEL PATHS
# ==========================================

CHAMPION_PATH = (
    BASE_DIR
    / "app"
    / "models"
    / "champion_model.pkl"
)


CHALLENGER_PATH = (
    BASE_DIR
    / "app"
    / "models"
    / "challenger_model.pkl"
)


# ==========================================
# LOAD MODELS
# ==========================================

champion_model = joblib.load(
    CHAMPION_PATH
)


challenger_model = joblib.load(
    CHALLENGER_PATH
)


# ==========================================
# PREDICT
# ==========================================

def predict(
    model_name,
    input_data
):

    df = pd.DataFrame([
        input_data
    ])


    if model_name == "champion":

        model = champion_model

    elif model_name == "challenger":

        model = challenger_model

    else:

        raise ValueError(
            "Invalid model name."
        )


    prediction = model.predict(df)[0]


    probability = model.predict_proba(
        df
    )[0][1]


    return (
        int(prediction),
        float(probability)
    )