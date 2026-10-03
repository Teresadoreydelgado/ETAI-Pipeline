"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
   load config -> load data -> clean -> split development/test
    -> split train/validation -> preprocess + train
    -> evaluate train/validation -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, build_preprocessor, split_dev_test 
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from sklearn.model_selection import train_test_split



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print(f"Model: {config['model']['type']} params={config['model'].get('params')}\n")

    # load + diagnose-and-clean (week 3): domain-rule/placeholder -> NaN, category
    # cleanup, de-duplication, redundant-column removal -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"]["id_column"])

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        config["preprocessing"]["mnar_indicator_sources"]
    )

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"]
    )

    # holdout: treino / validacao dentro do development set (o teste fica fechado)
    X_tr, X_va, y_tr, y_va, extras_tr, extras_va = train_test_split(
        X_dev,
        y_dev,
        extras_dev,
        test_size=0.25,
        random_state=config["test_set"]["random_state"],
        stratify=y_dev
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    pipeline.fit(X_tr, y_tr)

    y_tr_pred = pipeline.predict(X_tr)
    y_va_pred = pipeline.predict(X_va)

    report = evaluate(
        y_tr,
        y_tr_pred,
        y_va,
        y_va_pred
    )

    report += "\n" + fairness_report(
        y_va,
        y_va_pred,
        extras_va,
        sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
