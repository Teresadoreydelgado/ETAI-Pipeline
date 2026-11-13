"""Week 9: uses the final trained pipeline to predict outcomes for new data."""

import os

import pandas as pd

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new_data(pipeline, config: dict) -> str:
    """
    Predicts the file in `config["predict"]["input"]` with the fitted `pipeline` and saves
    id, probability and 0/1 prediction to `config["predict"]["output"]`. The new rows get
    the same cleaning (but never `drop_duplicate_rows`: every row needs a prediction) and the
    same feature split (there is no target). Returns the path of the saved file.
    """
    predict_config = config["predict"]
    df_new = clean_dataset(load_data(predict_config["input"]), config["diagnostics"])
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X_new, _, _ = split_features_target(df_new, config["data"], mnar_sources)

    proba = pipeline.predict_proba(X_new)[:, 1]
    threshold = predict_config.get("threshold", 0.5)
    id_column = config["diagnostics"].get("id_column", "id")

    predictions = pd.DataFrame({
        id_column: df_new[id_column].values,
        "probability": proba,
        "prediction": (proba >= threshold).astype(int),
    })

    if len(predictions) != len(df_new):
        raise ValueError(f"{len(df_new)} rows in, {len(predictions)} predictions out: every row needs a prediction")

    path = predict_config["output"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    predictions.to_csv(path, index=False)
    return path