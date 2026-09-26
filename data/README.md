20260670 Teresa Maria d'Orey Delgado


# WEEK 2 PROGRESS

## After EDA and Data Cleaning

This week focused on improving data quality by handling invalid values and placeholders, standardizing categories, removing duplicates, and dropping redundant features before using the existing preprocessing pipeline.

## Logistic Regression

Logistic Regression still generalizes well, with train accuracy of 0.673 and test accuracy of 0.669. The small gap (+0.004) indicates no clear overfitting. Compared with Week 1, performance remained broadly similar.

## Decision Tree

The Decision Tree reaches 0.795 train accuracy and 0.648 test accuracy, with a gap of +0.147. It still overfits, but less than in Week 1, when the gap was +0.202. Test accuracy also improved from 0.627 to 0.648.

## Overall comparison

Logistic Regression remains the more stable model, while the Decision Tree still overfits. However, the data cleaning appears to have improved the Decision Tree's generalization.

## Summing up

**Logistic Regression**

Week 1: train 0.679 | test 0.677 | gap +0.002  
Week 2: train 0.673 | test 0.669 | gap +0.004  

→ Essentially unchanged, still generalizing well.

**Decision Tree**

Week 1: train 0.829 | test 0.627 | gap +0.202  
Week 2: train 0.795 | test 0.648 | gap +0.147  

→ Better test performance and less severe overfitting.

# WEEK 1 PROGRESS

Logistic Regression

The Logistic Regression model shows good generalization, with very similar training and test accuracies (0.679 and 0.677), indicating no clear overfitting. Its test performance is moderate, and it performs better for class 0 than class 1, especially in terms of recall. The false positive rate also varies across racial groups, with a higher FPR for African-American than for Caucasian individuals.

Decision Tree

The Decision Tree achieves higher training accuracy (0.829) but lower test accuracy (0.627), with a large gap of 0.202, indicating clear overfitting. It also performs worse than Logistic Regression on class 1, with lower recall and F1-score. However, the FPR difference between the two largest racial groups is slightly smaller.

Overall comparison

Overall, Logistic Regression is the more reliable model because it generalizes better and achieves higher test performance. The Decision Tree fits the training data better but overfits and performs worse on unseen data. These results show the importance of evaluating models using test performance, class-specific metrics, and fairness measures rather than training accuracy alone.




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
