
# Chronic Disease Cost Forecasting

Predicts a Medicare beneficiary's next-year healthcare cost from their chronic
condition profile and prior spending, then sorts them into a Low, Medium, or
High risk tier. Built to reflect the kind of prospective risk modeling payer
analytics teams use to prioritize care management outreach.

**Live dashboard:** [Click Here!](https://chronic-disease-cost-forecasting.streamlit.app)

**One-line pitch:** A chronic disease cost forecasting model that segments
patients into cost trajectories, giving care teams a way to prioritize
intervention.

## Why this project

Managed care organizations that serve Medicaid and Medicare populations need
to know, ahead of time, which members are likely to become high-cost in the
coming year, so care management resources go to the people who need them
most. This project reproduces that workflow end to end: given a patient's
demographics, chronic conditions, and prior claims history, predict their
next-year cost and translate that prediction into an actionable risk tier.

## Data

CMS's 2008-2010 Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF),
Beneficiary Summary File, Sample 1. Free and publicly available from
[cms.gov](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf).

Sample 1 is one of 20 independent 0.25% random samples of the Medicare
population, roughly 116,000 beneficiaries. It was used in place of the full
file since it's already large enough for stable model training, and CMS's
synthesis method limits how much real-world accuracy additional samples
would add.

**Important limitation:** this is synthetic data. CMS explicitly states that
the relationships between variables have been partially altered to protect
beneficiary privacy, so results here should be read as a demonstration of
methodology, not as clinical or actuarial fact about the real Medicare
population.

## Methodology

### Framing: prospective risk modeling

Rather than predicting cost from same-year data, which would just be
correlation, this project uses 2008 and 2009 beneficiary data (demographics,
chronic conditions, prior cost) to predict 2010 total cost. This mirrors how
real payer risk models work: use what's known about a member today to
forecast their cost before the year starts.

### Data preparation

- Filtered to beneficiaries with full 12-month coverage in all three years
  (2008, 2009, 2010), 93,558 of 116,352 beneficiaries. Partial-year
  enrollees (death, plan switching) were excluded since their annual cost
  figures aren't comparable to full-year members.
- Target variable: `MEDREIMB_IP + MEDREIMB_OP + MEDREIMB_CAR` for 2010
  (inpatient, outpatient, and carrier/physician reimbursement).
- 11 rows had small negative cost values, likely claim adjustments in the
  synthetic data, and were floored to zero.
- Age range in the data extends below 65, consistent with Medicare
  eligibility rules for beneficiaries with disabilities or end-stage renal
  disease (ESRD).

### Target distribution

2010 cost is heavily right-skewed (mean $2,784, median $1,080, max
$116,910), typical of healthcare cost data. About 16% of beneficiaries had
zero cost in 2010, including 2,198 people with at least one chronic
condition flag, showing that a chronic condition does not guarantee high
utilization in a given year.

Linear and Ridge regression were trained on log-transformed cost
(`log1p`) to account for this skew. Random Forest and XGBoost were trained
on raw cost, since tree-based models don't require that assumption.

### Models compared

| Model | MAE | RMSE |
|---|---|---|
| Baseline (mean cost by chronic condition count) | $2,695.69 | $5,603.56 |
| Linear Regression | $2,491.35 | $5,976.78 |
| Ridge Regression | $2,491.34 | $5,976.78 |
| Random Forest | $2,571.19 | $5,524.42 |
| XGBoost | $2,567.49 | $5,511.41 |

XGBoost was selected as the final model. It's the only model to beat the
baseline on both metrics simultaneously, and it edges out Random Forest
slightly on both. The margin over Random Forest is modest, consistent with
the dataset's synthetic nature limiting how much additional signal is
available to extract. Training and test error were close (train MAE
$2,485.35 vs. test MAE $2,567.49), indicating the model is not overfitting.

Linear regression's better MAE but worse RMSE than the tree models reflects
a real tradeoff: the log-transform improves accuracy on typical patients but
increases error on high-cost outliers when converted back to dollars.

### Feature importance

Prior year cost (`cost_2008`, `cost_2009`, and their trend) accounts for
roughly 72% of the XGBoost model's decision-making, the strongest predictor
by a wide margin. This matches how real payer risk models are built, prior
utilization is typically the single best predictor of future cost.

The count of chronic conditions a patient has outweighs any individual
condition flag. Multimorbidity, not any specific diagnosis, is what drives
cost most, individual condition flags (diabetes, cancer, heart failure)
each contributed under 1% on their own.

### Risk tiers

Predicted cost is split into tiers by percentile of the predicted
distribution: bottom 50% (Low), next 35% (Medium), top 15% (High). Actual
average cost climbs cleanly across these tiers ($1,294 Low, $3,689 Medium,
$5,533 High), confirming the tiers meaningfully separate patients by real
spending.

In the test set, the High tier represents 15% of patients but accounts for
30% of total cost, a concentration of spending that justifies why targeted
outreach to a relatively small group can have an outsized impact.

## Limitations

- Built on synthetic data; absolute numbers should not be read as
  real-world Medicare cost figures.
- Trained on a 0.25% sample (Sample 1 of 20); the same pipeline would scale
  directly to the full file or to real claims data.
- Zero-inflation in the cost distribution (about 16% of patients) was
  identified but not modeled with a dedicated two-part (zero vs. non-zero)
  approach, a reasonable next step for further development.
- A modest decline in average population cost from 2008/2009 to 2010 was
  observed in this sample and not investigated further; it did not affect
  modeling decisions.

## Dashboard

Built in Streamlit, with two views:

- **Predict for a Patient**: enter a patient's demographics, chronic
  conditions, and 2008/2009 cost history (or select a preset example) to
  get a predicted 2010 cost and risk tier.
- **Population Overview**: shows how predicted risk tiers distribute across
  the test set and how actual cost varies by tier, including the cost
  concentration statistic above.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn,
Streamlit, deployed on Streamlit Community Cloud.

## Repository structure

```
chronic-cost-forecasting/
├── app.py                          Streamlit dashboard
├── data/
│   ├── raw/                        CMS SynPUF CSVs (not included, see Data section)
│   └── processed/
│       ├── processed_data.csv      Model-ready feature table
│       └── test_predictions.csv    Test set predictions for the population view
├── notebooks/
│   ├── 1_eda_and_feature_engineering.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── features.py                 Shared feature engineering logic
│   ├── predict.py                  Model loading and prediction logic
│   ├── xgb_cost_model.pkl          Trained XGBoost model
│   └── risk_tier_cutoffs.pkl       Risk tier percentile boundaries
└── README.md
```

## Running locally

```bash
git clone https://github.com/heyybaragi/chronic-cost-forecasting.git
cd chronic-cost-forecasting
pip install -r requirements.txt
streamlit run app.py
```

Raw CMS SynPUF files are not included in this repository due to size. Download
Sample 1 (Beneficiary Summary File, 2008-2010) from
[CMS's DE-SynPUF page](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf/de10-sample-1)
and place the three CSVs in `data/raw/` before running the notebooks.

## Author

Sneha Nannapaneni
[GitHub](https://github.com/heyybaragi)
