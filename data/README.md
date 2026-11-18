20260670 Teresa Maria d'Orey Delgado

# WEEK 10 PROGRESS

## Missing values and deployment

This week had two parts. First, the numeric imputer became a choice in `config.yaml` (`numeric_strategy`): median, mean, KNN, MICE (`IterativeImputer`) or missForest (`IterativeImputer` with a random forest). They all learn only from the features, never from the label, so they also work on new people. Numeric columns are now scaled before imputing, because KNN measures distances. Second, `main.py` now saves the final fitted pipeline to `models/model.joblib`, and a new `predict.py` loads it to predict new people without retraining. The test set remains locked.

### Pipeline progress

| Week | Pipeline improvements |
|---|---|
| W9 | Pipeline runs on a second dataset (Adult) with its own config: config file as an argument of `main.py`; general fairness report (reference score in the config); placeholders cleaned in every text column; predictions for new, label-free data (`src/predict.py`); `compare_runs.py` reads any scoreboard |
| W10 | Numeric imputer as a config option (median, mean, knn, iterative, random_forest), scaled before imputing; imputer shown in the terminal, the scoreboard and the report header; `main.py` saves the final pipeline (`models/model.joblib`); `predict.py` predicts a new CSV without retraining |

### Missing values

XGBoost untuned, feature selection on (`min_votes=2`), 5-fold CV on the development set. Only `numeric_strategy` changes between runs. Each imputer was compared with the median using the corrected paired t-test (`compare_runs.py`).

| Numeric imputer | ROC-AUC (mean ± std) | Train-validation gap | vs median (corrected t-test) |
|---|---|---|---|
| **median** | 0.706 ± 0.015 | +0.062 | - |
| knn | 0.707 ± 0.017 | +0.068 | +0.0006, wins 3/5, p = 0.80 |
| iterative (MICE) | 0.706 ± 0.016 | +0.070 | +0.0002, wins 2/5, p = 0.94 |
| random_forest (missForest) | 0.707 ± 0.014 | +0.070 | +0.0004, wins 3/5, p = 0.80 |

With the median, the result is exactly the same as in Week 9 (0.706 ± 0.015), so scaling before imputing changed nothing.

I keep the **median**. No imputer is really better (all differences below 0.001, all p ≥ 0.80), because the numeric imputer only sees the five numeric columns, which say little about each other, and the model only uses age and priors_count. Since the result is the same, the median is the better choice: it is the simplest and fastest, and it has the smallest train-validation gap.

### Predicting new data

```bash
python main.py                                                 # train, evaluate and save models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv   # predict new people, no retraining
```

`predict.py` loads the saved pipeline (the last run, with the median), cleans the new data with the same rules as training (without dropping duplicates, since every row needs an answer) and writes `predictions.csv`. It has one row per input row (20 in, 20 out), with three columns: `id`, `probability` (of reoffending) and `prediction` (1 if the probability is ≥ 0.5). The messy rows also get a prediction: a `?` in `priors_count` becomes missing and is imputed, an unseen charge degree (`X`) is accepted by the encoder, and the same person appearing twice gets the same prediction in both rows.

### Conclusions

- **Deleting rows is not an option in production:** a new person with an empty field still needs a prediction, so missing values have to be imputed.
- **A more complex imputer is not always better:** KNN, MICE and missForest did not beat the median, because an imputer is only as good as the columns it can see.
- **Predicting never retrains:** the saved pipeline carries everything learned in training (medians, encoder, selected features, model), so new data is only transformed and predicted.
- **Fairness did not change:** with every imputer, the false positive rate is still about 0.29 for African-American vs 0.18–0.19 for Caucasian defendants.

**Final reported result:** XGBoost (untuned) with feature selection and median imputation reached a ROC-AUC of 0.706 ± 0.015 using 2 features (age, priors_count). KNN, MICE and missForest gave the same result within noise (p ≥ 0.80).

# WEEK 9 PROGRESS

## Beyond COMPAS (option A: Adult income)

This week no new technique was added: the goal was to check whether the pipeline works on a second dataset. Option A uses the **Adult** dataset (predict whether a person earns more than 50K USD a year), which is still a yes/no question. Every part that broke showed a place in the code that only worked for COMPAS. Each fix had to be general: nothing specific to Adult is written in the code, everything dataset-specific lives in its own config file, and the COMPAS run still gives exactly the same numbers as in Week 8 (an untuned XGBoost run with selection gives 0.706 ± 0.015, as before, and `compare_runs.py` reproduces the Week 8 test: −0.0025, 1/5, p = 0.396).

### How to run

```bash
python main.py config_adult.yaml     # Adult (results in results_adult/)
python main.py                       # COMPAS, exactly as before (config.yaml by default)
python compare_runs.py results_adult/runs.csv   # corrected t-test on the Adult scoreboard
```

The predictions for the 2,000 new people are saved to `predictions/adult_new_predictions.csv` (`id`, probability, 0/1 prediction at threshold 0.5), made by the final model of the best run.

**Sensitive attribute:** `sex` is audited in the fairness report and is not a feature. `race` is also kept out of the features (in `drop_columns`): an income model should not decide based on someone's race, so neither sensitive attribute is used directly. They can still enter indirectly, for example through `relationship` (Husband/Wife) or `marital_status`. `fnlwgt` is dropped because it is a census sampling weight, not a characteristic of the person.

### Pipeline progress

| Week | Pipeline improvements |
|---|---|
| W8 | XGBoost; feature selection (`VoteSelector`) inside the pipeline, with `min_votes` tuned together with the model; cache folder; `compare_runs.py` (corrected paired t-test between two runs) |
| W9 | Pipeline runs on a second dataset (Adult) with its own config: config file as an argument of `main.py`; general fairness report (reference score in the config); placeholders cleaned in every text column; predictions for new, label-free data (`src/predict.py`); `compare_runs.py` reads any scoreboard |

### What broke, why, and how I fixed it

| What broke | Why | Fix |
|---|---|---|
| `KeyError: 'score_text'` in the fairness report | COMPAS's own score was written in the code (`fairness_report`, `split_features_target`) | `reference_score` moved to `config.yaml`; the report shows it only if the config names one (Adult has none) |
| `?` stayed as a category (no error!) | placeholders were only replaced in the columns listed in `canonical_categories`, which was enough for COMPAS | `clean_dataset()` now turns placeholders into NaN in every text column; the check after cleaning gives 379 / 381 / 86 missing (workclass / occupation / native_country) and no `?` category |
| `main.py` always read `config.yaml` | the config path was fixed | `python main.py <config>`, default `config.yaml` |
| no way to predict new data | the pipeline only evaluated | new `src/predict.py` + `predict` section in the config: same cleaning (without `drop_duplicate_rows`), rows out = rows in (2,000) |
| `compare_runs.py` only read the COMPAS scoreboard | the path was fixed | the scoreboard path is an optional first argument |

### Results (Adult)

5-fold CV on the development set (4,800 rows, 24% earn > 50K); nested CV for tuned runs. Test set locked.

| Run | ROC-AUC (mean ± std) | Accuracy | F1 | Train-validation gap | Features kept |
|---|---|---|---|---|---|
| Dummy (floor) | 0.500 ± 0.000 | 0.761 | 0.000 | +0.000 | - |
| Decision tree (default) | 0.724 ± 0.012 | 0.799 | 0.580 | +0.207 | all 36 |
| XGBoost (default) | 0.895 ± 0.015 | 0.847 | 0.645 | +0.105 | all 36 |
| XGBoost tuned, no selection | 0.907 ± 0.016 | **0.858** | **0.662** | +0.023 | all 36 |
| **XGBoost tuned + selection** | **0.908 ± 0.014** | **0.857** | **0.673** | +0.013 | 8: age, education_num, capital_gain, capital_loss, hours_per_week, marital_status, occupation, relationship |
| Target | ≥ 0.90 | ≥ 0.855 | ≥ 0.66 | | |

Chosen hyperparameters: without selection n_estimators=180, learning_rate=0.033, max_depth=4; with selection min_votes=1, n_estimators=522, learning_rate=0.115, max_depth=1.

Corrected paired t-test (with − without selection, ROC-AUC): mean difference +0.0015, wins 4/5 folds, t = 1.20, p = 0.296.

The hardest assumption to find was the placeholder cleaning, because it failed silently: the `?` in `workclass`, `occupation` and `native_country` stayed as a category, the pipeline ran without any error, and only the check after cleaning showed it. The `score_text` error was easy to find, because the program stopped and named the column. What reached the target scores was tuning, not a different model: the default XGBoost overfitted (gap +0.105) and missed all three targets, while the tuned one chose shallower trees and a lower learning rate, reduced the gap to +0.023 and reached all three. Feature selection was not needed for the targets, since the model without it already reached them, and it did not change the ROC-AUC in a real way (p = 0.30), but it reached the same score with 8 of 36 features and discarded all 25 undocumented columns (v01–v25) as noise.

### Conclusions

- **The pipeline is now general.** The same code runs both datasets, each with its own config, and COMPAS gives exactly the same numbers as before.
- **Accuracy alone is misleading here.** Always predicting "≤ 50K" already gives 0.761, because only 24% of people earn more; that is why the targets also include ROC-AUC and F1.
- **Tuning fixed overfitting.** XGBoost went from a gap of +0.105 to +0.023 and from 0.895 to 0.907 ROC-AUC. Compared with the defaults (depth 6, learning rate 0.3), tuning chose shallower trees (depth 4, and depth 1 with selection) and a lower learning rate.
- **Feature selection behaves as in COMPAS.** In both datasets it does not change the ROC-AUC in a real way (p = 0.40 for COMPAS, p = 0.30 for Adult), but it gives a much smaller model (2 of 10 and 8 of 36 features).
- **Fairness:** in every run that predicts anyone as > 50K, the false positive rate is higher for men than for women (final model 0.09 vs 0.03): men who earn ≤ 50K are wrongly predicted as > 50K about three times as often. Tuning without selection gave the lowest rates (0.08 vs 0.01); feature selection raised recall and with it the FPR of both groups slightly, but no run removed the difference between them.

**Final reported result:** XGBoost tuned with feature selection reached a nested CV ROC-AUC of 0.908 ± 0.014, accuracy 0.857 and F1 0.673, meeting all three targets with 8 of 36 features. It is not significantly different from the model with all 36 features (0.907 ± 0.016, p = 0.30), which also meets the targets. The nested CV score is reported instead of the best Optuna trial score, as it provides a more reliable estimate of generalization performance.

# WEEK 8 PROGRESS

## XGBoost and Feature Selection

This week, two changes were made to the pipeline. First, **XGBoost** was added as a new model: it is gradient boosting with regularisation inside each split (`gamma`, `min_child_weight`, `reg_lambda`, `reg_alpha`). Second, a **feature selection** step (`VoteSelector`) was added inside the pipeline, between preprocessing and the model. Three models vote on each feature: a random forest and gradient boosting vote for features with above-average importance, and an L1-penalised logistic regression votes for features whose coefficient is not shrunk to zero. A feature is kept if it gets at least `min_votes` votes.

Because the selection looks at the labels, it is fitted only on the training rows of each fold, like every other learned step; selecting features on all rows before cross-validation would leak information and give an optimistic score. `min_votes` is tuned together with the model (`select__min_votes`). A cache folder (`memory=`) stores the fitted preprocessing and selection steps, so they are not refitted in every tuning trial; this changes the running time, not the results.

XGBoost was tuned with and without feature selection (30 Optuna trials, nested CV, same folds as Week 7), and the two runs were compared with the corrected paired t-test. The test set remains locked.

### Pipeline progress

| Week | Pipeline improvements |
|---|---|
| W7 | Gradient Boosting, HistGradientBoosting and Naive Bayes; 8 evaluation metrics; ROC-AUC selection; out-of-fold probabilities; persistent scoreboard |
| W8 | XGBoost; feature selection (`VoteSelector`) inside the pipeline, with `min_votes` tuned together with the model; cache folder; `compare_runs.py` (corrected paired t-test between two runs) |

### Scoreboard

Selection metric: **ROC-AUC**. Results are reported as mean ± standard deviation across 5 folds, using nested CV for tuned models. The no-model floor is 0.500 for all models.

| Model | Untuned ROC-AUC | Tuned ROC-AUC (mean ± std) | Floor | Chosen hyperparameters |
|---|---|---|---|---|
| Gradient Boosting | 0.727 | 0.728 ± 0.014 | 0.500 | n_estimators=307, learning_rate=0.0292, max_depth=2, min_samples_leaf=1, subsample=0.7346 |
| HistGradientBoosting | 0.717 | 0.728 ± 0.015 | 0.500 | max_iter=82, learning_rate=0.1109, max_leaf_nodes=4, min_samples_leaf=130, l2_regularization=0.000700 |
| **XGBoost** | 0.706 | 0.727 ± 0.016 | 0.500 | n_estimators=451, learning_rate=0.0165, max_depth=7, min_child_weight=3.58, subsample=0.552, gamma=2.49 |
| Oblique Random Forest | 0.686 | 0.727 ± 0.015 | 0.500 | tree max_depth=13, min_samples_leaf=91, n_components=8 |
| AdaBoost | 0.723 | 0.727 ± 0.014 | 0.500 | base tree max_depth=4, n_estimators=55, learning_rate=0.1674 |
| Random Forest | 0.683 | 0.726 ± 0.013 | 0.500 | max_depth=9, min_samples_leaf=42, max_features=0.5 |
| **XGBoost + feature selection** | 0.706 | 0.725 ± 0.013 | 0.500 | min_votes=3, n_estimators=273, learning_rate=0.0366, max_depth=7, min_child_weight=1.31, subsample=0.637, gamma=2.79 |
| Logistic Regression | 0.719 | Not tuned | 0.500 | max_iter=2000, class_weight=balanced |
| Decision Tree | 0.605 | 0.718 ± 0.015 | 0.500 | max_depth=9, min_samples_leaf=100, criterion=gini |
| Naive Bayes | 0.685 | 0.705 ± 0.012 | 0.500 | var_smoothing=0.09115 |
| Dummy | 0.500 | Not tuned | 0.500 | Default parameters |

### Feature selection (XGBoost, tuned, ROC-AUC, nested CV)

| Run | Untuned | Tuned (nested) mean ± std | Train-validation gap (tuned) | Features kept |
|---|---|---|---|---|
| Without selection | 0.706 | 0.727 ± 0.016 | +0.013 | all 10 |
| With selection (VoteSelector, min_votes=3) | 0.706 | 0.725 ± 0.013 | +0.006 | 2: age, priors_count |

Corrected paired t-test (with − without): mean difference −0.0025, wins 1/5 folds, t = −0.95, p = 0.396.

The smaller model is as good as the full one: with only 2 of the 10 features, the tuned XGBoost loses 0.0025 of ROC-AUC, wins in 1 of 5 folds, and the corrected t-test gives p = 0.40, so there is no real difference. The selection kept only age and the number of prior offences, and the choice of `min_votes` hardly matters: on COMPAS, no other feature gets votes, so 1, 2 or 3 votes keep the same two features. I would keep the selection on, because a smaller model that scores the same is the better model: it is simpler, faster, easier to explain, and it overfits less (gap +0.006 vs +0.013, with a smaller std across folds).

### Model comparison

Untuned, XGBoost was one of the weaker models (0.706) and overfitted the most among the boosting models (gap +0.083), because its defaults are aggressive (depth 6, learning rate 0.3). After tuning, it reached 0.727 ± 0.016, tied with the best Week 7 models (Gradient Boosting and HistGradientBoosting, 0.728). Tuning chose a small learning rate and a high `gamma` (about 2.5), which blocks splits that barely improve the fit, and the gap dropped to +0.013.

With feature selection, XGBoost scored 0.725 ± 0.013 using only 2 features, within the noise of every top model.

### Conclusions

- **XGBoost needs tuning.** Its aggressive defaults overfit (gap +0.083); tuning brought the score from 0.706 to 0.727 and the gap down to +0.013, mainly by choosing a low learning rate and strong split regularisation (`gamma`).
- **The strongest models still perform similarly.** Tuned XGBoost (0.727) ties with Gradient Boosting and HistGradientBoosting (0.728). As in previous weeks, more model complexity does not produce a meaningful gain.
- **Two features carry almost all the information.** Age and number of prior offences give the same ROC-AUC as all 10 features (p = 0.40), which explains why every model reaches a similar ceiling: the limit is the information in the data, not the model.
- **Feature selection must be inside the cross-validation.** It learns from the labels, so it is refitted on the training rows of each fold; otherwise it would leak information and give an optimistic score.
- **Fairness did not change.** In both runs, the false positive rate is still about twice as high for African-American (0.30) as for Caucasian (0.16) defendants: lower than COMPAS's own score (0.45 vs 0.23), but neither XGBoost nor feature selection reduced the disparity.

**Final reported result:** XGBoost with feature selection achieved a nested CV ROC-AUC of 0.725 ± 0.013 using 2 features, not significantly different from the 10-feature model (0.727 ± 0.016, p = 0.40). The nested CV score is reported instead of the best Optuna trial score, as it provides a more reliable estimate of generalization performance.

# WEEK 7 PROGRESS

## Boosting and Evaluation Metrics

This week, three new models were added to the pipeline: Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes. The evaluation was extended from accuracy alone to eight metrics: accuracy, balanced accuracy, precision, recall, F1, F-beta (beta=2), ROC-AUC and average precision.

The pipeline now uses out-of-fold predicted probabilities to calculate these metrics and generate classification and fairness reports. ROC-AUC was selected as the optimization metric for Optuna and for ranking models in the new scoreboard (`results/runs.csv`).

All nine models were evaluated using stratified 5-fold cross-validation, with nested cross-validation for tuned models (30 Optuna trials). The 20% test set remains locked and was not used for evaluation or model selection.

### Pipeline progress

| Week | Pipeline improvements |
|---|---|
| W7 | Gradient Boosting, HistGradientBoosting and Naive Bayes; 8 evaluation metrics; ROC-AUC selection; out-of-fold probabilities; persistent scoreboard |

### Scoreboard

Selection metric: **ROC-AUC**. Results are reported as mean ± standard deviation across 5 folds, using nested CV for tuned models. The no-model floor is 0.500 for all models.

| Model | Untuned ROC-AUC | Tuned ROC-AUC (mean ± std) | Floor | Chosen hyperparameters |
|---|---|---|---|---|
| Gradient Boosting | 0.727 | 0.728 ± 0.014 | 0.500 | n_estimators=307, learning_rate=0.0292, max_depth=2, min_samples_leaf=1, subsample=0.7346 |
| HistGradientBoosting | 0.717 | 0.728 ± 0.015 | 0.500 | max_iter=82, learning_rate=0.1109, max_leaf_nodes=4, min_samples_leaf=130, l2_regularization=0.000700 |
| Oblique Random Forest | 0.686 | 0.727 ± 0.015 | 0.500 | tree max_depth=13, min_samples_leaf=91, n_components=8 |
| AdaBoost | 0.723 | 0.727 ± 0.014 | 0.500 | base tree max_depth=4, n_estimators=55, learning_rate=0.1674 |
| Random Forest | 0.683 | 0.726 ± 0.013 | 0.500 | max_depth=9, min_samples_leaf=42, max_features=0.5 |
| Logistic Regression | 0.719 | Not tuned | 0.500 | max_iter=2000, class_weight=balanced |
| Decision Tree | 0.605 | 0.718 ± 0.015 | 0.500 | max_depth=9, min_samples_leaf=100, criterion=gini |
| Naive Bayes | 0.685 | 0.705 ± 0.012 | 0.500 | var_smoothing=0.09115 |
| Dummy | 0.500 | Not tuned | 0.500 | Default parameters |

### Selection metric

ROC-AUC was chosen because the COMPAS task involves ranking individuals by predicted risk of recidivism, rather than only making binary classifications. Unlike accuracy, ROC-AUC evaluates how well the model distinguishes between the two classes across different classification thresholds, without depending on a fixed threshold of 0.5. It is therefore more appropriate for comparing the models' ability to rank individuals by risk.

### Model comparison

The best results were achieved by Gradient Boosting and HistGradientBoosting, both with a ROC-AUC of 0.728. Oblique Random Forest, AdaBoost and Random Forest followed closely, with scores between 0.726 and 0.727.

Hyperparameter tuning substantially improved the Decision Tree (0.605 to 0.718), Random Forest (0.683 to 0.726) and Oblique Random Forest (0.686 to 0.727). In contrast, Gradient Boosting performed almost equally well with its default configuration (0.727 vs 0.728).

The differences between the five best models are very small relative to their standard deviations, so the results do not establish a clear winner.

### Conclusions

- **ROC-AUC provides a more appropriate evaluation for this task.** It measures the quality of risk ranking independently of the classification threshold, while the remaining metrics provide complementary information about classification performance.
- **Tuning improves some models more than others.** The largest improvements occurred in the Decision Tree and the two random forests, while Gradient Boosting already performed well before tuning.
- **The strongest models perform similarly.** The five best models achieved ROC-AUC values between 0.726 and 0.728, suggesting that additional model complexity does not necessarily produce a meaningful improvement.
- **The no-model floor confirms that the models learn useful patterns.** All eight non-dummy models exceeded the Dummy ROC-AUC of 0.500.
- **The test set remains untouched.** All comparisons use cross-validation on the development set, preserving the locked test set for future evaluation.

**Final reported result:** Gradient Boosting achieved a nested CV ROC-AUC of 0.728 ± 0.014, tied with HistGradientBoosting at the reported precision. The nested CV score is reported instead of the best Optuna trial score, as it provides a more reliable estimate of generalization performance.

**Note:** Week 7 ROC-AUC results are not directly comparable with the accuracy scores reported in Weeks 5 and 6, because they measure different aspects of predictive performance.


# WEEK 6 PROGRESS

## Ensembles: Random Forest, Oblique Random Forest and AdaBoost

This week the pipeline gained two new model types in `src/model.py`, built from scikit-learn parts: an **oblique random forest** (each tree splits on random linear combinations of the features, so its splits are slanted lines instead of steps; the trees are combined with a `BaggingClassifier`) and **AdaBoost** over shallow decision trees. The random forest now uses a fixed 200 trees, because more trees never overfit, they only cost time; its tuning controls the size of the trees instead. All three ensembles were tuned with the Week 5 setup (Optuna, 30 trials, nested CV), on the same folds, with the test set still locked.

### Ensembles

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | - | - | - |
| Decision tree | 0.611 ± 0.015 (+0.084) | 0.675 ± 0.020 (+0.009) | 0.680 | max_depth 9, min_samples_leaf 101, gini |
| Random forest | 0.652 ± 0.019 (+0.081) | 0.678 ± 0.014 (+0.011) | 0.683 | max_depth 22, min_samples_leaf 42, max_features 0.5 |
| Oblique random forest | 0.650 ± 0.009 (+0.113) | 0.682 ± 0.014 (+0.011) | 0.683 | tree max_depth 4, min_samples_leaf 2, n_components 10 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.016 (+0.005) | 0.684 | base tree max_depth 4, n_estimators 220, learning_rate 0.025 |

**Progress across weeks**

| Model | Week | Evaluation | Train accuracy | Test / Validation accuracy | Std (across folds) | Gap |
|---|---|---|---|---|---|---|
| Dummy | 4 | Holdout | 0.549 | 0.550 | - | ~0.000 |
| Dummy | 5 | 5-fold CV | 0.549 | 0.549 | 0.000 | ~0.000 |
| Logistic regression | 2 | Test set | 0.679 | 0.677 | - | +0.002 |
| Logistic regression | 3 | Test set | 0.673 | 0.669 | - | +0.004 |
| Logistic regression | 4 | Holdout | 0.679 | 0.674 | - | +0.005 |
| Logistic regression (one-hot encoding) | 4 | Holdout | 0.675 | 0.660 | - | +0.015 |
| Logistic regression | 5 | 5-fold CV | 0.675 | 0.672 | 0.013 | +0.003 |
| Logistic regression (tuned) | 5 | Nested CV | 0.677 | 0.673 | 0.016 | +0.004 |
| Decision tree | 2 | Test set | 0.829 | 0.627 | - | +0.202 |
| Decision tree | 3 | Test set | 0.795 | 0.648 | - | +0.147 |
| Decision tree | 4 | Holdout | 0.679 | 0.609 | - | +0.070 |
| Decision tree | 5 | 5-fold CV | 0.695 | 0.611 | 0.015 | +0.084 |
| Decision tree (max_depth 5) | 5 | 5-fold CV | 0.685 | 0.675 | 0.018 | +0.011 |
| Decision tree (tuned) | 5 | Nested CV | 0.684 | 0.675 | 0.020 | +0.009 |
| Random forest | 4 | Holdout | 0.727 | 0.656 | - | +0.072 |
| Random forest (max_depth 5) | 4 | Holdout | 0.690 | 0.679 | - | +0.011 |
| Random forest | 5 | 5-fold CV | 0.732 | 0.652 | 0.017 | +0.080 |
| Random forest (tuned) | 5 | Nested CV | 0.688 | 0.679 | 0.014 | +0.009 |
| Random forest (200 trees) | 6 | 5-fold CV | 0.732 | 0.652 | 0.019 | +0.081 |
| Random forest (200 trees, tuned) | 6 | Nested CV | 0.689 | 0.678 | 0.014 | +0.011 |
| Oblique random forest | 6 | 5-fold CV | 0.763 | 0.650 | 0.009 | +0.113 |
| Oblique random forest (tuned) | 6 | Nested CV | 0.693 | 0.682 | 0.014 | +0.011 |
| AdaBoost | 6 | 5-fold CV | 0.682 | 0.676 | 0.012 | +0.006 |
| AdaBoost (tuned) | 6 | Nested CV | 0.687 | 0.682 | 0.016 | +0.005 |

### Model comparison

After tuning, the three ensembles are essentially tied (0.678-0.682), only slightly above the tuned Decision Tree (0.675) and Logistic Regression (0.672). The default forests overfit strongly (gaps of +0.081 and +0.113), while AdaBoost with stumps barely overfits even before tuning.

### Conclusions

- **No model is clearly the best.** The tuned ensembles differ by at most 0.004, and even their lead over Logistic Regression (about 0.010) is smaller than the std across folds (0.014-0.016). With these 10 features, every model ends up around 0.68: the limit seems to be the information in the data, not the model.
- **The chosen hyperparameters match how each model works.** AdaBoost fixes bias by adding weak trees one after another, so it chose shallow trees (depth 4) with a small learning rate and many rounds. The random forest fixes variance by averaging independent trees, so it chose deep trees (depth 22), but with large leaves (at least 42 people) so that no tree memorizes individuals. The oblique forest chose shallow trees (depth 4): each oblique split combines several features, so fewer splits are needed.
- **Overfitting in a forest comes from the size of the trees, not their number.** With 200 unlimited trees, both forests overfit; limiting leaf size (random forest) or depth (oblique forest) removed most of the gap.
- **The model and number I would report: AdaBoost, with a nested CV accuracy of 0.682 ± 0.016.** It is tied for the best score, has the smallest gap, and was already good with default settings. The best-trial score (0.684) is not reported, because it is the maximum over 30 trials and therefore optimistic. Since its lead is within the noise, Logistic Regression would be a reasonable simpler choice.
- **Fairness did not change with the ensembles.** The false positive rate is still about twice as high for African-American (0.29-0.30) as for Caucasian (0.15-0.17) defendants. This is lower than COMPAS's own score (0.45 vs 0.23), but more complex models did not reduce the disparity.

# WEEK 5 PROGRESS

## Part 1: Cross-validation

### Model evaluation

A single train/validation split is not reliable, because the score depends on which people happen to fall in the validation set. This week the single train/validation split was replaced by stratified 5-fold cross-validation on the development set. Each model is trained on 4 folds and validated on the remaining one, 5 times, and the result is reported as mean ± standard deviation. Preprocessing is refit inside each fold, so there is no leakage, and the test set (20%) stays locked.

| Model | Holdout accuracy (W4) | CV accuracy (mean ± std) | CV train-val gap |
|---|---|---|---|
| Dummy | 0.550 | 0.549 ± 0.000 | -0.000 |
| Logistic regression | 0.674 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.609 | 0.611 ± 0.015 | +0.084 |
| Decision tree (max_depth 5) | - | 0.675 ± 0.018 | +0.011 |
| Random forest | 0.656 | 0.652 ± 0.017 | +0.080 |

**Progress across weeks**

| Model | Week | Evaluation | Train accuracy | Test / Validation accuracy | Std (across folds) | Gap |
|---|---|---|---|---|---|---|
| Dummy | 4 | Holdout | 0.549 | 0.550 | - | ~0.000 |
| Dummy | 5 | 5-fold CV | 0.549 | 0.549 | 0.000 | ~0.000 |
| Logistic regression | 2 | Test set | 0.679 | 0.677 | - | +0.002 |
| Logistic regression | 3 | Test set | 0.673 | 0.669 | - | +0.004 |
| Logistic regression | 4 | Holdout | 0.679 | 0.674 | - | +0.005 |
| Logistic regression (one-hot encoding) | 4 | Holdout | 0.675 | 0.660 | - | +0.015 |
| Logistic regression | 5 | 5-fold CV | 0.675 | 0.672 | 0.013 | +0.003 |
| Logistic regression (tuned) | 5 | Nested CV | 0.677 | 0.673 | 0.016 | +0.004 |
| Decision tree | 2 | Test set | 0.829 | 0.627 | - | +0.202 |
| Decision tree | 3 | Test set | 0.795 | 0.648 | - | +0.147 |
| Decision tree | 4 | Holdout | 0.679 | 0.609 | - | +0.070 |
| Decision tree | 5 | 5-fold CV | 0.695 | 0.611 | 0.015 | +0.084 |
| Decision tree (max_depth 5) | 5 | 5-fold CV | 0.685 | 0.675 | 0.018 | +0.011 |
| Decision tree (tuned) | 5 | Nested CV | 0.684 | 0.675 | 0.020 | +0.009 |
| Random forest | 4 | Holdout | 0.727 | 0.656 | - | +0.072 |
| Random forest (max_depth 5) | 4 | Holdout | 0.690 | 0.679 | - | +0.011 |
| Random forest | 5 | 5-fold CV | 0.732 | 0.652 | 0.017 | +0.080 |
| Random forest (tuned) | 5 | Nested CV | 0.688 | 0.679 | 0.014 | +0.009 |

### Model comparison

The CV results are close to the Week 4 holdout results. Logistic Regression is still the most stable model, with the smallest gap and the lowest std. The Decision Tree with max_depth 5 performs about the same, while the default Decision Tree and the Random Forest still overfit, which is why they score lower.

### Conclusions

- **The CV number is more trustworthy than the holdout number.** With a single split, the score depends on which people happen to fall in the validation set. CV uses every row of the development set for validation once and averages 5 scores, and the std shows how much the result can change just by changing the split.
- **Small differences between models are not meaningful.** The std across folds is around 0.013-0.018, so a difference like 0.672 vs 0.675 (Logistic Regression vs Decision Tree with max_depth 5) is within the noise. Before CV, we could not tell this.
- **The conclusion from previous weeks holds, with one nuance.** Logistic Regression is still the most reliable model and generalizes well. But a Decision Tree with limited depth is just as good, which shows that the tree's problem in earlier weeks was overfitting, not the model itself.
- **The gap is what reveals overfitting.** The default Decision Tree and the Random Forest have large train-validation gaps, because without a depth limit the trees memorize the training data. For example, the same Decision Tree goes from 0.611 (no depth limit) to 0.675 (max_depth 5), a difference much larger than the std. Their hyperparameters were chosen by hand (or left as default), so the next step is to tune them properly (Part 2).
- **The Dummy confirms the models are learning something.** It always predicts the majority class and gets 0.549 in every fold, so every real model clearly beats this baseline.

## Part 2: Hyperparameter tuning

### Hyperparameter tuning

Hyperparameters (like a tree's max_depth) are not learned by `fit()`, so they have to be chosen by comparing candidates. The pipeline uses Optuna (TPE sampler, 30 trials): each trial picks one value per hyperparameter from the search space in `config.yaml`, and the whole pipeline (preprocessing + model) is scored with 5-fold CV on the development set. Because the best of 30 trials is optimistic (it partly won by luck), the reported number comes from nested CV: the tuning is repeated inside each of the 5 outer folds, and the chosen model is scored on the outer fold that took no part in the choice. The test set stays locked, so there is no leakage.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.611 ± 0.015 | 0.675 ± 0.020 | 0.680 | -0.000 | max_depth 9, min_samples_leaf 101, gini |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | C = 0.072 |
| Random forest | 0.652 ± 0.017 | 0.679 ± 0.014 | 0.682 | +0.003 | n_estimators 117, max_depth 8, min_samples_leaf 15, max_features sqrt |

### Model comparison

Tuning helped the two tree models a lot and Logistic Regression almost not at all. After tuning, the three models score very close to each other (0.673-0.679), and the train-validation gaps are all small (+0.004 to +0.009).

### Conclusions

- **Tuning only helps when the default overfits.** The Decision Tree gained +0.064 and the Random Forest +0.027, both clearly larger than the std across folds, because tuning limited their depth and leaf size, and their gaps dropped from about +0.08 to +0.009. Logistic Regression gained only +0.001, much smaller than the std: it has one hyperparameter and was not overfitting to begin with.
- **Tuning did not create a clear winner.** The Random Forest is now numerically the best (0.679), but the differences between the three tuned models (at most 0.006) are smaller than the std, so they are effectively tied. The Random Forest went from the worst model to one of the best, which shows that its earlier results reflected untuned hyperparameters, not a weak model.
- **The number to report is the nested CV, not the best trial.** The best trial is the maximum over 30 candidates, so it is optimistic. Here the optimism is very small (-0.000 to +0.003), so the tuning did not fool itself, but the honest number is still the nested CV one: 0.679 ± 0.014 for the Random Forest.
- **The chosen hyperparameters vary from fold to fold** (e.g. the tree's max_depth ranges from 5 to 13), but the scores are consistent. Many combinations are almost equally good, so what matters is the score, not the exact values.

# WEEK 4 PROGRESS

## After Preprocessing

This week introduced imputation, encoding and scaling inside the model pipeline, together with a separate validation set and a locked test set.

| Model | Week | Train accuracy | Test / Validation accuracy | Gap |
|---|---|---|---|---|
| Dummy | 4 | 0.549 | 0.550 | ~0.000 |
| Logistic regression | 2 | 0.679 | 0.677 | +0.002 |
| Logistic regression | 3 | 0.673 | 0.669 | +0.004 |
| Logistic regression | 4 | 0.679 | 0.674 | +0.005 |
| Logistic regression (one-hot encoding) | 4 | 0.675 | 0.660 | +0.015 |
| Decision tree | 2 | 0.829 | 0.627 | +0.202 |
| Decision tree | 3 | 0.795 | 0.648 | +0.147 |
| Decision tree | 4 | 0.679 | 0.609 | +0.070 |
| Random forest | 4 | 0.727 | 0.656 | +0.072 |
| Random forest (max_depth 5) | 4 | 0.690 | 0.679 | +0.011 |

Weeks 2 and 3 were evaluated on the test set; from Week 4 the test set is locked and models are evaluated on a validation set.

## Summing up

**Logistic Regression** → Performance remained very stable, with good generalization.

**Decision Tree** → The overfitting gap decreased, although validation performance was lower.

**Random Forest** → Limiting tree depth reduced overfitting and improved validation performance.  

## Model comparison

Logistic Regression showed the most stable performance, with a very small gap, and remained essentially unchanged across weeks.

The Decision Tree and Random Forest showed larger train-validation gaps, indicating more overfitting. Random Forest performed better on validation than the single Decision Tree. Compared with previous weeks, the Decision Tree's overfitting gap kept decreasing, although its validation accuracy was lower.

The Dummy model achieved around 0.55 accuracy, providing a baseline for comparison.

## Pipeline changes

Since Logistic Regression showed the most stable baseline performance, it was used to test an encoder change. Replacing target encoding with one-hot encoding reduced validation accuracy and increased the gap, so target encoding was kept.

Random Forest was also tested with `max_depth: 5`. Validation accuracy improved and the gap decreased a lot, suggesting that limiting tree depth reduces overfitting.


# WEEK 3 PROGRESS

## After EDA and Data Cleaning

This week focused on improving data quality by handling invalid values and placeholders, standardizing categories, removing duplicates, and dropping redundant features before using the existing preprocessing pipeline.

| Model | Week | Train accuracy | Test accuracy | Gap |
|---|---|---|---|---|
| Logistic regression | 2 | 0.679 | 0.677 | +0.002 |
| Logistic regression | 3 | 0.673 | 0.669 | +0.004 |
| Decision tree | 2 | 0.829 | 0.627 | +0.202 |
| Decision tree | 3 | 0.795 | 0.648 | +0.147 |

**Logistic Regression** still generalizes well. The small gap indicates no clear overfitting, and performance remained broadly similar to Week 2.

**Decision Tree** still overfits, but less than in Week 2. Test accuracy also improved, and the gap decreased.

**Overall comparison:** Logistic Regression remains the more stable model, while the Decision Tree still overfits. However, the data cleaning appears to have improved the Decision Tree's generalization.

# WEEK 2 PROGRESS

| Model | Train accuracy | Test accuracy | Gap |
|---|---|---|---|
| Logistic regression | 0.679 | 0.677 | +0.002 |
| Decision tree | 0.829 | 0.627 | +0.202 |

**Logistic Regression** shows good generalization, with very similar training and test accuracies, indicating no clear overfitting. Its test performance is moderate, and it performs better for class 0 than class 1, especially in terms of recall. The false positive rate also varies across racial groups, with a higher FPR for African-American than for Caucasian individuals.

**Decision Tree** achieves higher training accuracy but lower test accuracy, with a large gap, indicating clear overfitting. It also performs worse than Logistic Regression on class 1, with lower recall and F1-score. However, the FPR difference between the two largest racial groups is slightly smaller.

**Overall comparison:** Logistic Regression is the more reliable model because it generalizes better and achieves higher test performance. The Decision Tree fits the training data better but overfits and performs worse on unseen data. These results show the importance of evaluating models using test performance, class-specific metrics, and fairness measures rather than training accuracy alone.



# Dataset -- COMPAS Recidivism (ProPublica)

## The problem

In 2016, ProPublica investigated COMPAS, a risk-assessment algorithm
actually used by courts in Broward County, Florida, to help inform
bail and sentencing decisions. COMPAS scores a defendant's likelihood
of reoffending on a 1-10 scale; judges could see that score when
deciding, among other things, whether someone should be released
before trial. ProPublica obtained COMPAS's scores for thousands of
defendants and matched them against what actually happened over the
following two years, then published the data.

This dataset is that data: each row is one defendant, with their
demographics and criminal history at the time of screening, COMPAS's
own risk score for them, and whether they were actually rearrested
within two years.

**Your task:** predict `two_year_recid` -- will this person be
rearrested within two years? -- from the case facts. Once you have a
model, the more interesting question is the one ProPublica actually
asked: is it equally accurate for everyone, or does it get things
wrong more often, in a particular direction, for some groups than
others? `race` is deliberately excluded from the model's own inputs
(see `config.yaml` and `src/preprocessing.py`) so it can be used
afterward purely to check this, in `src/evaluate.py`.

Before any of that: look at the data first. It comes from a real
system with real data-entry and record-keeping quirks -- don't assume
every column is clean or consistent just because it loads without
error.

## Data dictionary

| column | type | description | notable values |
|--------|------|--------------|------------------|
| `id` | identifier | internal record id | not a model feature |
| `sex` | categorical | defendant's sex | `Male`, `Female` |
| `age` | numeric | defendant's age (years) at screening | |
| `age_cat` | categorical | age bucket | `Less than 25`, `25 - 45`, `Greater than 45` |
| `race` | categorical | defendant's race, as recorded | `African-American`, `Caucasian`, `Hispanic`, `Asian`, `Native American`, `Other`; excluded from model features, used only to audit fairness |
| `juv_fel_count` | numeric | number of prior juvenile felony offenses | |
| `juv_misd_count` | numeric | number of prior juvenile misdemeanor offenses | |
| `juv_other_count` | numeric | number of other prior juvenile offenses | |
| `juvenile_total` | numeric | total juvenile offenses | |
| `priors_count` | numeric | number of prior adult offenses | |
| `prior_offenses` | numeric | number of prior offenses | |
| `age_in_months` | numeric | age expressed in months | |
| `c_charge_degree` | categorical | degree of the current charge | `F` (felony), `M` (misdemeanor) |
| `decile_score` | numeric | COMPAS's own risk score | 1 (lowest risk) to 10 (highest risk); excluded from model features, used only for comparison |
| `score_text` | categorical | COMPAS's own risk category | `Low`, `Medium`, `High`; excluded from model features, used only for comparison |
| `two_year_recid` | binary | **target** -- was this person rearrested within two years? | `0` = no, `1` = yes |

Source: derived from [propublica/compas-analysis](https://github.com/propublica/compas-analysis) (the data behind the "Machine Bias" investigation). Personally-identifying columns (name, date of birth, case numbers, charge descriptions) were removed.
