# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml            # all tunable settings live here
├── compare_runs.py        # corrected paired t-test between two runs of the scoreboard
├── requirements.txt
├── src/
│   ├── data.py            # loading
│   ├── preprocessing.py   # cleaning, preprocessing + development/test split
│   ├── model.py           # models, feature selection (VoteSelector) and build_pipeline()
│   ├── tuning.py          # Optuna tuning + nested cross-validation
│   ├── evaluate.py        # metrics, cross-validation, fairness check, corrected t-test
│   └── results.py         # saves each run's report and the scoreboard (runs.csv)
├── results/               # created automatically -- one report per run + runs.csv
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md          # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA & data cleaning | `clean_dataset()`: placeholder tokens and invalid values become NaN, category spellings standardized, duplicate rows and redundant columns removed |
| 4 | Preprocessing inside the pipeline | Imputation, encoding (target encoder) and scaling inside an sklearn `Pipeline`, missingness indicators, a locked test set and a validation split; dummy and random forest models added |
| 5 | Cross-validation & hyperparameter tuning | Stratified 5-fold CV replaces the single split; Optuna tuning (`src/tuning.py`) with nested CV for an honest estimate; `build_pipeline()` in `src/model.py` |
| 6 | Ensembles | Oblique random forest and AdaBoost added to `src/model.py`; random forest fixed at 200 trees; search spaces for the three ensembles in `config.yaml` |
| 7 | Boosting & evaluation metrics | Gradient boosting, HistGradientBoosting and naive Bayes added; 8 evaluation metrics computed from out-of-fold probabilities, with ROC-AUC as the selection metric; persistent scoreboard of all runs (`results/runs.csv`); corrected paired t-test (`corrected_paired_test()` in `src/evaluate.py`) |
| 8 | XGBoost & feature selection | XGBoost added to `src/model.py`; feature selection (`VoteSelector`) as a learned step inside the pipeline, between preprocessing and the model, with `min_votes` tuned together with the model; cache folder (`memory=`) so tuning doesn't refit preprocessing and selection in every trial; `compare_runs.py` compares two runs with the corrected t-test |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and cleans the data, locks the test set, and evaluates the pipeline (preprocessing, optional feature selection and model) with stratified 5-fold cross-validation on the development set. It prints:
- **the selection metric (ROC-AUC) with its train-validation gap and the no-model floor.** A large gap means the model memorised the training folds instead of learning something that generalises; the floor (a dummy model) shows what "no learning" scores.
- if tuning is enabled: the honest nested CV score of the tuned model and the chosen hyperparameters
- all 8 metrics (mean ± std across folds)
- the scoreboard of all runs (`results/runs.csv`), best first
- the features the final model keeps

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
