# Explaining Instant EMI-Finance Decisions (Responsible & Safe AI, AIML4001 – Set B)

An explainability pipeline for an NBFC's instant EMI-loan model. A Random Forest
predicts credit risk on the **German Credit (UCI Statlog)** dataset. **SHAP** shows
what drives the model overall, and **SHAP + LIME** explain one rejected applicant.
The pipeline also writes a plain-language rejection message and flags sensitive
attributes (age, personal_status, foreign_worker).

## Files

| Path | Purpose |
|---|---|
| `main.py` | Complete program (tasks 1-6) |
| `main.ipynb` | Same code as a Jupyter notebook, with outputs and plots |
| `data/german_credit.csv` | 1,000 applicants, 20 attributes, `class` = good / bad |
| `outputs/` | `shap_summary.png`, `shap_waterfall.png`, `lime_explanation.png` |
| `RSA_SetB_Report.pdf` | Report: aim, algorithm, program, output, answers to Q1-Q4, conclusion |

## Run

```bash
pip install -r requirements.txt
python main.py
```

Settings: `random_state = 42`, OrdinalEncoder (one feature per attribute), stratified
80:20 split, "bad" = positive class (1), RandomForest(n_estimators=200, max_depth=8),
LIME `num_features = 5`.

## Results

```
Model : RandomForest (200 trees, depth 8)
Test accuracy : 0.77      ROC-AUC : 0.83

Top-5 mean |SHAP| (class = bad):
 1 checking_status 0.092 | 2 duration 0.042 | 3 credit_amount 0.027
 4 savings_status 0.027  | 5 purpose 0.025

Applicant (test index 23) : P(bad) = 0.57 -> REJECTED
SHAP top 3 : checking_status=<0 (+0.11), duration=24 (+0.06),
             personal_status=female div/dep/mar (+0.04)
LIME top 5 : checking_status=<0 (+0.12), other_payment_plans=none (-0.06),
             personal_status=female div/dep/mar (+0.04), age <= 27 (+0.04),
             3 < installment_commitment <= 4 (+0.03)
Top-3 agreement : 2 / 3
Sensitive attributes in reasons : age, personal_status -> REVIEW
Message: "Your EMI request was not approved mainly because your current account
balance is negative and the requested duration (24 months) is long."
```

Dataset source: Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine
Learning Repository. https://doi.org/10.24432/C5NC77
