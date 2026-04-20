# Brief

Paste verbatim host material here. Do not paraphrase — invariances and
constraints often hide in the wording.

## Competition description

Welcome to the 2026 Kaggle Playground Series! We plan to continue in the
spirit of previous playgrounds, providing interesting and approachable
datasets for our community to practice their machine learning skills, and
anticipate a competition each month.

Your Goal: Predict the irrigation need.

## Rules

**Competition**: Playground Series - Season 6, Episode 4
(`playground-series-s6e4`)
**Sponsor**: Google LLC
**Prizes**: Choice of Kaggle merchandise (no cash)
**Winner license**: None
**Data license**: CC BY 4.0
**Status at kickoff**: 10 days to go (per competition page banner)

### Key limits

- **Team size**: max 3
- **Daily submissions**: 10
- **Final submissions selected for judging**: up to 2
- **Leaderboard**: Public LB scored on a "representative sample" of test;
  **Private LB** determines final standing (Playground typically ~50/50
  split — confirm later from discussion forum if relevant).

### External data & tools

- External data **allowed** provided it's publicly available and equally
  accessible to all participants at minimal cost. The original Irrigation
  Prediction dataset (the one the synthetic data was generated from) is
  explicitly allowed by the data description.
- External models: allowed unless specifically prohibited. AutoML tools
  (AMLT) explicitly permitted.
- Open source code OK under OSI-approved licenses that don't restrict
  commercial use.

### Prohibited

- **Hand labeling or human prediction of test/validation data** is
  explicitly forbidden (§ 3.4.b). No label leakage via manual annotation.
- Private code/data sharing outside registered teams.
- Multiple accounts.

## Evaluation

Submissions are evaluated on **balanced accuracy** between the predicted
class and observed target.

### Submission File

For each `id` in the test set, you must predict a class label
(`Low`, `Medium`, `High`) for the `Irrigation_Need` target. The file should
contain a header and have the following format:

```
id,Irrigation_Need
630000,Low
630001,High
630002,Low
etc.
```

## Data description

The dataset for this competition (both train and test) was generated from a
deep learning model trained on the Irrigation Prediction dataset. Feature
distributions are close to, but not exactly the same as, the original. Feel
free to use the original dataset as part of this competition, both to explore
differences as well as to see whether incorporating the original in training
improves model performance.

### Files

- **train.csv** — the training set, with `Irrigation_Need` as target
- **test.csv** — the test set, used to predict the category for
  `Irrigation_Need`
- **sample_submission.csv** — a sample submission file in the correct format
  <remaining bullet cut off in screenshot — to confirm>

### Columns

From inspecting `train.csv` header + first row:

Target: `Irrigation_Need` ∈ {`Low`, `Medium`, `High`}

Features (19):

| Column | Type | Example |
|---|---|---|
| `id` | int (identifier, drop for modeling) | 0 |
| `Soil_Type` | categorical | Loamy |
| `Soil_pH` | numeric | 4.92 |
| `Soil_Moisture` | numeric | 32.58 |
| `Organic_Carbon` | numeric | 1.01 |
| `Electrical_Conductivity` | numeric | 3.05 |
| `Temperature_C` | numeric | 15.01 |
| `Humidity` | numeric | 50.61 |
| `Rainfall_mm` | numeric | 725.99 |
| `Sunlight_Hours` | numeric | 5.9 |
| `Wind_Speed_kmh` | numeric | 16.79 |
| `Crop_Type` | categorical | Sugarcane |
| `Crop_Growth_Stage` | categorical | Sowing |
| `Season` | categorical | Zaid |
| `Irrigation_Type` | categorical | Drip |
| `Water_Source` | categorical | Rainwater |
| `Field_Area_hectare` | numeric | 0.82 |
| `Mulching_Used` | categorical (Yes/No) | No |
| `Previous_Irrigation_mm` | numeric | 112.16 |
| `Region` | categorical | East |

Dataset sizes:
- `train.csv`: 630,000 rows (id 0 – 629,999)
- `test.csv`: 270,000 rows (id 630,000 – 899,999)
- Ratio ≈ 70/30 train/test

### Class distribution (train)

| Class | Count | Fraction |
|---|---|---|
| Low | 369,917 | **58.7%** |
| Medium | 239,074 | **37.9%** |
| High | 21,009 | **3.3%** |

**Severely imbalanced.** Under balanced accuracy (macro-recall), `High` is
worth 1/3 of the score on 1/30 of the data — so threshold tuning /
class-balanced sample weights / focal loss on `High` is the biggest
expected lever.

## Host forum / notebook comments

<none yet>

## Flagged invariances / constraints

- **Task**: 3-class classification. Classes `Low`, `Medium`, `High` — has
  natural ordinal structure but the metric does not reward ordinal
  closeness, so off-by-one errors (High→Low) count the same as (High→Medium).
- **Metric**: balanced accuracy = macro-averaged recall over the 3 classes.
  Implications:
  - Class imbalance in training does not help or hurt the metric directly;
    what matters is per-class recall. Upweighting rare classes / class-
    balanced sample weights / threshold tuning on predicted probs can all
    move the score meaningfully.
  - Default `argmax` is rarely optimal under balanced accuracy when classes
    are imbalanced — tune class-specific thresholds or decision rules on
    OOF predictions to maximize macro-recall.
  - Metric is bounded [0, 1]; random 3-class guessing scores ~0.333.
- **Submission format**: hard string labels (`Low`/`Medium`/`High`), not
  probabilities. Check exact capitalization before submitting.
- **Synthetic data**: train/test generated by a DL model trained on the
  original **Irrigation Prediction** dataset. Distributions close to but
  **not identical** to original → original dataset is explicitly allowed as
  additional training data, but its DGP may diverge. Worth testing as a
  separate experiment rather than blindly concatenating.
- **Test id range**: the example starts at `630000`, implying a training
  set of roughly 630k rows and a separate test block starting at 630000.
  Confirm exact sizes once data is downloaded.
- **Submission budget**: 10/day × ~10 days = ~100 total submissions
  remaining at kickoff. Spend them on high-information-gain probes (e.g.
  testing whether the original dataset helps, or whether threshold tuning
  matters) rather than re-confirming strong leaders.
- **Public LB is a "representative sample"** → trust CV when they disagree,
  but large public/private gaps in Playground series are rare. Still, do
  not overfit to public LB via probe submissions.
- **No hand labeling of test data** (§ 3.4.b) — rules out any "manually
  inspect test rows then special-case them" shortcuts.
