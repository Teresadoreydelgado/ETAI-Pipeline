"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> split development/test (test set locked)
    -> stratified k-fold CV on the development set -> (optional) hyperparameter tuning
    -> reports -> refit final model -> save results
"""

import yaml
from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test 
from src.model import build_pipeline
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.results import save_run
from sklearn.model_selection import StratifiedKFold
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report



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

    # preprocessing + model = one estimator
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # week 5 (part 1): stratified k-fold CV on the development set
    cv_config = config["cv"]

    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle, random_state=cv_config.get("random_state") if shuffle else None)

    scoring = cv_config.get("scoring", "accuracy")

    n_jobs = cv_config.get("n_jobs", 1)

    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)
    report = "Default hyperparameters\n" + cv_report(fold_scores, scoring)

    # week 5: hyperparameter tuning -- development set only, every candidate scored by
    # CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but there is no search space for '{model_type}' in "
                "config.yaml (tuning.search_spaces). Add one or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True,
        random_state=tuning_seed)
        # 1. the honest estimate (its out-of-fold predictions replace the untuned ones)
        print("\nNested cross-validation (tuned model)")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed,
            n_jobs=n_jobs,
            )
        report += "\n\nNested cross-validation (tuned model)\n" + cv_report(nested_scores, scoring)
        
        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        print()
        pipeline, study = tune_pipeline(
        pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed,
        n_jobs=n_jobs,
        )
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    print()   
    report += "\n\n" + oof_classification_report(y_dev, y_oof)

    report += "\n" + fairness_report( y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    # the final model: the pipeline refit on all development rows -- test set still locked
    final_model = pipeline.fit(X_dev, y_dev)

    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
