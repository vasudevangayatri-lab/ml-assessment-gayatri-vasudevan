# Part B — Business Case Analysis
## Scenario: Promotion Effectiveness at a Fashion Retail Chain

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations.
Each month the marketing team runs one of five promotions across stores that vary in
size, footfall, competition density, and customer demographics. The objective is to
determine which promotion to deploy in each store each month to maximise items sold.

- **Five promotions:** Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. 
- **Stores diffractor:** Vary in size, monthly footfall, local competition density, and customer demographics. 


---

## B1. Problem Formulation

---

### *B1(a) — ML Problem Formulation*

#### Target Variable

**`items_sold`** — the number of units sold per store per month under a given promotion.

This is a continuous, non-negative integer that is directly observable at the end of
each promotional period and is well-defined regardless of price changes, discounting
depth, or product mix shifts.

#### Candidate Input Features

| Feature category | Example features | Rationale |
|---|---|---|
| **Store characteristics** | `store_size`, `location_type`, `store_id` | Fixed structural attributes that set the baseline sales ceiling |
| **Temporal** | `month`, `year`, `day_of_week`, `is_weekend`, `is_festival`, `is_month_end` | Capture seasonality, payday effects, and festive demand spikes |
| **Competitive context** | `competition_density` | Local competitive pressure shapes promotional responsiveness |
| **Promotion** | `promotion_type` (the decision variable) | The controllable lever the retailer optimises each month |
| **Historical performance** | Rolling average `items_sold` per store, promotion lift ratios from prior months | Encode store-level baseline and promotion-specific track record |
| **Customer demographics** | Average customer age band, income index per catchment area | Explain why the same promotion lands differently across locations |

#### Type of ML Problem

This is a **supervised regression problem with a downstream optimisation objective**.

- **Regression** because the target (`items_sold`) is a continuous numeric quantity,
  not a class label.
- **Supervised** because labelled historical records (store × month × promotion →
  items_sold) exist and are used to fit the model.
- **Optimisation objective** because the trained model is used as a *scoring function*:
  for each store and each upcoming month, the model predicts `items_sold` under each
  of the five promotion types, and the promotion with the highest predicted score is
  selected. This makes the overall system a **counterfactual regression / uplift
  modelling** framework — the regression model answers "what would sales be *if* we
  ran promotion X here?"

**Justification for regression over classification:**
A classification framing (predict *which* promotion wins) discards the magnitude of
the difference between options. Knowing that BOGO is predicted to outsell Flat Discount
by 120 units vs 3 units is critical for prioritisation and budget allocation decisions.
Regression preserves this ordinal and cardinal information.

---

### *B1(b) — Why Items Sold Is a More Reliable Target Than Revenue*

#### The case against revenue as a target

Sales revenue is the product of two quantities: **units sold × selling price**. In a
fashion retail context, price varies continuously because:

- **Promotional mechanics directly manipulate price:** a Flat Discount reduces the
  per-unit price; a BOGO halves the effective price per item; a Category-Specific
  Offer applies variable depth discounts across a subset of SKUs.
- **Product mix shifts under promotions:** a Free Gift promotion may drive customers
  toward high-margin accessories to qualify for the gift, inflating revenue without
  a true volume increase.
- **Seasonal markdowns interact with promotions:** if a store runs a 40% end-of-season
  markdown *and* a loyalty points campaign simultaneously, the revenue signal reflects
  the markdown depth as much as the promotion's effectiveness.

When the promotion itself changes the price, revenue becomes **circular** as a target
— we would be partly measuring the input (price reduction) as if it were the output
(commercial performance). A model trained on revenue would systematically recommend
deeper discounts simply because they mechanically produce the highest revenue numbers
at sufficient volume, not because they are truly the best promotion strategy.

#### Why items sold is cleaner

`items_sold` is **price-invariant**. Whether a unit sells at full price, at 20%
discount, or as the second item in a BOGO deal, it counts as one. This means:

- The model learns which promotion drives genuine **demand lift** — additional customers
  entering the store or existing customers buying more units — rather than which
  promotion inflates a price-contaminated number.
- Cross-promotion comparisons are **fair**: BOGO and Flat Discount are evaluated on
  equal footing (units moved) rather than on the different revenue bases their pricing
  mechanics create.
- The signal is **operationally actionable**: the warehouse, supply chain, and
  merchandising teams all plan around units, not revenue. A model that predicts units
  aligns directly with the decisions these teams must make.

#### The broader principle: target variable alignment

This illustrates the principle of **target variable alignment** — the chosen target
must measure the causal outcome the business cares about, free from confounding
factors introduced by the very inputs being optimised.

A corollary is the distinction between **proxies and true outcomes**. Revenue is a
proxy for commercial success, but it is a *price-contaminated* proxy when price is
itself a feature of the interventions that is being compared. Items sold is closer to the
true causal outcome — demand — and is therefore a more robust and interpretable
optimisation target. In ML projects, by choosing a target that is correlated
with the intervention mechanism instead of the cause of this, may be a common
sources of misleading models and low quality deployment outcomes.

---

### *B1(c) — Alternative to a Single Global Model*

#### The problem with one global model across all 50 stores

A single global model implicitly assumes that the relationship between features and
`items_sold` is the same everywhere. In this retail context that assumption is
strongly violated:

- An urban store with high footfall and dense competition responds to BOGO very
  differently from a rural store with a loyal, low-competition customer base where
  Loyalty Points may be far more effective.
- Festival-season lift in a semi-urban market town may be three times the lift in
  a metropolitan store where customers have many alternative options.
- Customer price sensitivity — the key mechanism behind Flat Discount and BOGO
  effectiveness — varies systematically with location income levels.

Pooling all 50 stores forces the model to learn a single average response surface
that fits no store particularly well.

#### Proposed alternative: Hierarchical (Mixed-Effects) Modelling with Store Segments

The recommended strategy has two complementary components:

**1. Segment stores into groups before modelling**

Cluster the 50 stores along structural dimensions — `location_type`, `store_size`,
and `competition_density` — into 3–5 homogeneous segments (e.g. using K-Means, as
demonstrated in Q2). Train a **separate regression model per segment**. Stores within
a segment share similar baseline demand and promotional responsiveness, so each model
learns a more accurate local response surface than any global model could.

**2. Incorporate store-level random effects (hierarchical / mixed-effects model)**

Within each segment, use a **hierarchical model** (e.g. a Linear Mixed-Effects Model
or a Bayesian hierarchical regression) that includes:
- **Fixed effects:** features that apply uniformly across all stores in the segment
  (month, promotion type, is_festival, competition_density)
- **Random effects (store-level intercepts and slopes):** per-store deviations from
  the segment average, capturing idiosyncratic store behaviour without overfitting
  to any individual store's limited history

This structure — sometimes called *partial pooling* — is the key advantage over both
extremes:

| Approach | Problem |
|---|---|
| One global model | Underfits heterogeneous stores — ignores local variation |
| One model per store | Overfits to small per-store sample sizes — poor generalisation |
| **Hierarchical model (proposed)** | Borrows statistical strength across stores while preserving store-level variation |

#### Practical deployment

For the monthly promotion decision, the pipeline would be:

1. Look up the store's segment assignment (fixed, updated quarterly).
2. Run the segment's model to score all five promotion types for the upcoming month's
   context (month, festival flags, competition density).
3. Select the promotion with the highest predicted `items_sold`.
4. After the month ends, feed the actual result back into the model as a new training
   observation — enabling continuous online learning as new store-level data accumulates.

This strategy is operationally viable (five segment models rather than 50 individual
ones), statistically sound (partial pooling preventing overfitting), and respects the
structural heterogeneity that makes a single global model inappropriate.

---

## B2. Data and EDA Strategy

---

### *B2(a) — Joining Four Source Tables*

#### Source table schemas

| Table | Natural key(s) | Key columns |
|---|---|---|
| `transactions` | `transaction_id` | `store_id`, `date`, `promotion_id`, `items_sold`, `revenue` |
| `store_attributes` | `store_id` | `store_size`, `location_type`, `monthly_footfall`, `competition_density` |
| `promotion_details` | `promotion_id` | `promotion_type`, `discount_depth`, `mechanic_description` |
| `calendar` | `date` | `is_weekend`, `is_festival`, `month`, `year`, `day_of_week` |

#### Join sequence and logic

The joins are performed in a star-schema pattern with `transactions` as the central
fact table. All joins are **left joins from transactions outward** so that every
transaction record is retained regardless of whether lookup tables have a matching row
(missing lookups are flagged and investigated rather than silently dropped).

```
transactions
  LEFT JOIN store_attributes   ON transactions.store_id    = store_attributes.store_id
  LEFT JOIN promotion_details  ON transactions.promotion_id = promotion_details.promotion_id
  LEFT JOIN calendar           ON transactions.date         = calendar.date
```

**Key decisions:**

- **`store_attributes` is time-invariant** in this dataset. If store size or location
  type ever changes historically (e.g. a store was reclassified from semi-urban to
  urban mid-period), a date-ranged join (`store_id` AND `effective_date BETWEEN
  valid_from AND valid_to`) would be required to avoid attribute leakage.

- **Null `promotion_id` rows** in `transactions` represent the 80% of non-promoted
  transactions (addressed fully in B2(c)). These are retained with `promotion_type =
  'none'` after the join, not dropped, since no-promotion baseline behaviour is
  essential context for measuring lift.

- **Calendar join is on exact date**, producing one row per transaction date with
  all temporal flags attached. Festival dates that span multiple days each receive
  the same flag.

#### Grain of the final modelling dataset

> **One row = one store × one calendar month × one promotion type**

The transaction-level grain is too fine for monthly planning — the business decision
is made once per store per month. The modelling grain is therefore the
**store-month-promotion combination**.

This requires the following aggregations from transaction level up to modelling grain:

| Aggregation | Source column | Output feature | Purpose |
|---|---|---|---|
| `SUM` | `items_sold` | `items_sold` *(target)* | Total units moved in the store-month under the given promotion |
| `SUM` | `revenue` | `total_revenue` | Reference metric (not the target, but useful for sanity checks) |
| `COUNT` | `transaction_id` | `transaction_count` | Footfall proxy — number of customer visits |
| `MEAN` | `items_sold` per transaction | `avg_basket_units` | Average units per visit — separates volume effect from frequency effect |
| `MAX` | `is_festival` | `any_festival_day` | Binary: did the month contain at least one festival day? |
| `SUM` | `is_festival` | `festival_day_count` | Finer-grained than binary — a month with 3 festival days behaves differently from one |
| `MAX` | `is_weekend` | — | Already implicit in month-level aggregation; retained at day level if sub-monthly models are used |

**Additional engineered aggregations:**

- **Rolling 3-month average `items_sold` per store** — captures store-level trend
  and baseline without leaking future information (computed strictly from months
  prior to the current row).
- **Promotion lift ratio** — `items_sold` in promoted months divided by the rolling
  no-promotion baseline for the same store. This becomes both a feature (past lift)
  and a validation metric (does the model predict lift directionally correctly?).
- **Promotion frequency per store** — how often each promotion type has been run
  historically per store. Rarely-run promotions have sparse training data and may
  require shrinkage or hierarchical pooling.

The final modelling table has one row per store-month-promotion observation. For
months where only one promotion was run, there is one row per store per month. If the
dataset contains experiments where multiple promotions were tested simultaneously in
different store zones, the grain remains store-zone × month × promotion.

---

### *B2(b) — EDA Strategy Before Modelling* 

The following four analyses are performed in sequence, each informing a specific
modelling or feature engineering decision.

---

#### Analysis 1 — Target Distribution and Store-Level Baseline Spread

**Chart:** Histogram of `items_sold` at the store-month grain, overlaid with a KDE
curve. Alongside it, a box plot of `items_sold` by `store_id` sorted by median.

**What to look for:**

- **Skewness or multimodality** in the overall distribution. A right-skewed target
  suggests a log-transform may stabilise variance and improve linear model fit
  (`log(items_sold + 1)` as target).
- **Outlier months** — individual store-months with items_sold far beyond the
  interquartile range indicate data quality issues (double-counted transactions,
  extraordinary events) or genuine extreme periods that need careful handling.
- **Store-level spread** in the box plot. If the interquartile ranges of different
  stores barely overlap, it confirms that `store_id` or store structural features
  are dominant predictors, and store-level fixed effects or random intercepts are
  essential. If the ranges overlap heavily, a global model may be more viable.

**Modelling influence:** Extreme right skew → log-transform the target. Wide
between-store spread → include store-level random effects or segment-level models
(supporting B1(c) recommendation).

---

#### Analysis 2 — Promotion Lift by Type and Location

**Chart:** Grouped bar chart of mean `items_sold` per `promotion_type`, indicated by
`location_type` (urban / semi-urban / rural). Overlay error bars (± 1 standard
deviation or 95% confidence interval).

**What to look for:**

- **Promotion × location interaction effects.** If BOGO is the top promotion in
  urban stores but Loyalty Points dominates in rural stores, a global model that
  cannot capture this interaction will give systematically wrong recommendations.
  The presence of strong interactions is the empirical justification for location-
  segmented models.
- **Whether any promotion is universally dominant.** If one promotion type ranks
  first in every location, the optimisation problem becomes trivial — but this is
  unlikely in practice.
- **Statistical uncertainty** (wide error bars for some promotion-location cells)
  flags sparse data. Cells with fewer than, say, 10 store-month observations should
  be treated with caution and may benefit from hierarchical shrinkage rather than
  relying on their raw means.

**Modelling influence:** Significant interaction effects → include
`promotion_type × location_type` interaction terms or train separate models per
location segment. Sparse cells → apply hierarchical pooling or regularisation to
prevent overfitting to small samples.

---

#### Analysis 3 — Temporal Patterns and Seasonality

**Chart:** Line chart of monthly mean `items_sold` aggregated across all stores, with
a secondary axis showing `festival_day_count` per month as a bar overlay. Separately,
a heat map of day-of-week × month mean `items_sold` (7 × 12 grid).

**What to look for:**

- **Seasonal trend:** Do sales peak in specific months (festive quarter, summer)?
  A clear seasonal pattern means `month` is a strong feature and may benefit from
  cyclical encoding (sine/cosine transformation of month number) rather than treating
  it as a raw integer, so the model understands that month 12 and month 1 are
  temporally adjacent.
- **Festival amplification:** Does the festival day count track closely with sales
  peaks? If yes, `festival_day_count` is more informative than the binary
  `any_festival_day` flag and should replace or supplement it.
- **Year-on-year trend:** A consistently rising or falling baseline across 2022–2024
  means `year` or a trend index feature is needed. If growth is non-linear,
  a log-linear trend term or spline may be required.
- **Within-week rhythm** in the heat map: if Saturdays and festival Mondays behave
  distinctly from other days, sub-monthly models or day-of-week fixed effects add value.

**Modelling influence:** Cyclical month pattern → sine/cosine encoding of `month`.
Festival count > binary flag → replace `is_festival` with `festival_day_count`.
Year-on-year trend → include `year` or an auto-regressive trend feature.

---

#### Analysis 4 — Competition Density vs Promotion Effectiveness

**Chart:** Scatter plot of `competition_density` (x-axis) vs promotion lift ratio
(y-axis: promoted months' sales / no-promotion rolling baseline), coloured by
`promotion_type`, with a LOWESS smoothing line per promotion type.

**What to look for:**

- **Whether lift varies monotonically with competition density.** If BOGO lift
  declines as competition density increases (customers have more alternatives and
  compare prices), but Loyalty Points lift increases (customers prefer to accumulate
  points at a trusted store rather than switch), this is a critical feature
  interaction for store-level promotion assignment.
- **Non-linearity:** If the LOWESS line curves or has an inflection point, a linear
  model will underfit this relationship. A polynomial term or tree-based model is
  better suited.
- **Threshold effects:** If lift is roughly constant below `competition_density = 3`
  and drops sharply above it, a binary threshold feature (`high_competition = 1 if
  density ≥ 3`) may be more useful than the raw continuous value.

**Modelling influence:** Non-linear competition effect → use tree-based model or add
polynomial/interaction terms. Promotion × competition interaction → include
`promotion_type × competition_density` interaction or treat `competition_density` as
a stratification variable.


---

### *B2(c) — Addressing the 80% No-Promotion Imbalance* 

#### How the imbalance affects the model

The dataset has an 80/20 split between no-promotion and promoted transactions. This
creates two distinct problems:

**1. Biased coefficient estimation in regression models.**
If the model is trained on a dataset where 80% of rows have `promotion_type = 'none'`,
the coefficients for promotion features are estimated from only 20% of the data. The
model becomes well-calibrated for the no-promotion baseline but unreliable for
predicting differential lift across the five promotion types — precisely the
comparison the business needs.

**2. Suppressed promotion signal in feature importance.**
Tree-based models (Random Forest, Gradient Boosting) determine split quality by the
weighted average reduction in impurity across all rows. Because 80% of rows see no
promotion effect, promotion-type splits appear less informative than they truly are
for the 20% of promoted observations. Promotion features may be ranked artificially
low, masking their true decision-relevance.

**3. Baseline dominance in error metrics.**
A model that ignores promotion type entirely and simply predicts the store-month
baseline will score well on RMSE because 80% of rows (no promotion) are well-
explained by store-level intercepts and temporal patterns. The model appears accurate
while completely failing at the actual task — ranking promotions.

#### Steps to address the imbalance

**Step 1 — Reframe the target as promotion lift, not raw items_sold.**
Compute `lift = items_sold / rolling_baseline_items_sold` where the baseline is the
rolling 3-month no-promotion average for that store. Modelling lift instead of raw
sales removes the baseline from the target, making the model focus entirely on the
incremental promotion effect. This is the most structurally correct fix because it
aligns the target directly with the decision being optimised.

**Step 2 — Train on promoted observations only for the promotion-selection model.**
Separate the dataset into two sub-models:
- A **baseline model** trained on no-promotion rows to predict the store-month
  baseline (controls for seasonality, store size, competition density).
- A **lift model** trained exclusively on the 20% promoted rows to predict
  promotion-specific lift given store and temporal context.
The final prediction is `predicted_items_sold = baseline × predicted_lift`, which
cleanly separates the two estimation problems.

**Step 3 — Apply sample weights if a single model is required.**
If a unified model is preferred for simplicity, up-weight promoted observations by a
factor of 4 (inverse of their 0.20 frequency, normalised) during training. In
scikit-learn this is achieved via the `sample_weight` parameter in `model.fit()`.
This rebalances the loss function so that errors on promoted rows carry equal total
weight to errors on no-promotion rows.

**Step 4 — Use stratified cross-validation during model selection.**
When comparing models via cross-validation, stratify folds by `promotion_type`
(treating 'none' as one stratum) to ensure every fold has a representative mix of
promoted and non-promoted observations. Without stratification, some folds may
contain very few promoted rows, producing misleadingly high CV scores driven
entirely by baseline prediction accuracy.

**Broader principle:** This is an instance of the **data imbalance problem** applied
to regression rather than classification. The solution in both settings is the same:
reframe the target to isolate the signal of interest, train on the relevant
sub-population, and ensure the evaluation metric measures what the business actually
cares about — here, the ability to rank five promotions correctly, not the ability to
predict non-promoted sales.


---

## B3. Model Evaluation and Deployment

---

### *B3(a) — Train-Test Split, Metrics, and Interpretation* 

#### Setting up the train-test split

The dataset spans **36 months × 50 stores**, giving approximately 1,800 store-month
observations at the modelling grain (before filtering for promoted rows). The split
must respect chronological order for the same reasons established in Q3 Task 2: the
business deployment scenario is always to predict the *future* from the *past*, and
any split that allows future data into training constitutes leakage.

The recommended approach is a **walk-forward (rolling-origin) evaluation** rather than
a single static split:

```
Year 1–2 (months 1–24)   →  Initial training window
Month 25                  →  First test month
                              Retrain on months 1–25, predict month 26
                              Retrain on months 1–26, predict month 27
                              ... continue to month 36
```

This generates **12 out-of-sample test months** (months 25–36, the entirety of year 3),
each evaluated on predictions made using only data available at that point in time.
Walk-forward evaluation is superior to a single 80/20 cut because:

- It produces **12 independent error estimates** rather than one, enabling confidence
  intervals around the metrics.
- It tests the model at different points in the seasonal cycle — a single cut at
  month 29 might happen to land in a low-volatility period and produce optimistically
  narrow error estimates.
- It simulates the actual deployment loop: train → predict → observe → retrain.

**Why a random split is inappropriate** here extends the argument made in Q3 beyond
simple leakage. In a panel dataset (50 stores × 36 months), random splitting also
breaks **within-store temporal autocorrelation** — consecutive months for the same
store share promotional carry-over effects, seasonal baselines, and trend momentum.
Splitting randomly places store 12's December 2023 in training and its November 2023
in test, allowing the model to see the "future" December in training while evaluating
it on the "past" November — a nonsensical causal reversal. It also violates the
assumption of independent and identically distributed test observations that most
metric confidence intervals rely on.

#### Evaluation metrics and business interpretation

**1. RMSE (Root Mean Squared Error) — operational planning precision**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

RMSE is reported in the same units as `items_sold`. For a store averaging 300 units
per month, an RMSE of 25 means the model's predictions are typically within ±25 units
— roughly 8% of the mean. RMSE's quadratic penalty means large individual errors (e.g.
catastrophically mispredicting a festival month) are weighted heavily. This makes it
the right primary metric for **supply chain planning**, where large stockout errors are
disproportionately costly.

**2. MAE (Mean Absolute Error) — median store-month accuracy**

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i - y_i|$$

MAE is more robust than RMSE to the occasional extreme month (major festival,
unexpected stockout). It answers the question: "For a typical store in a typical
month, how wrong is the recommendation?" MAE is the metric to report to the marketing
team in day-to-day performance reviews because it is directly interpretable: "On
average, the model's promotion choice leads to a prediction error of 20 units."

**3. Promotion Ranking Accuracy — the business-critical metric**

Neither RMSE nor MAE directly measures whether the model picks the *right* promotion.
A model that predicts 310 units for BOGO and 305 for Flat Discount when the true
values are 312 and 290 respectively is directionally correct (BOGO wins) despite
having non-zero errors. The metric that captures this is:

$$\text{Ranking Accuracy} = \frac{\text{months where recommended promotion = highest actual promotion}}{\text{total test months}}$$

Computed per store and aggregated, this directly answers the business question: "What
fraction of the time does the model correctly identify the best promotion?" A 70%
ranking accuracy means the model outperforms random selection (20% for five options)
by a factor of 3.5 — a concrete, marketable figure for the executive audience.

**4. Counterfactual Lift — business value realised**

After deployment, compare `items_sold` in months where the model's recommendation was
followed against the historical baseline for the same store and same month of year.
This is the ultimate validation metric because it measures **actual business impact**
rather than prediction accuracy. A 12% average lift over the no-model baseline
directly quantifies the model's commercial value in monetary terms.


---

### *B3(b) — Investigating and Communicating Differing Recommendations*

#### Why the same store gets different recommendations in different months

The model recommends Loyalty Points for Store 12 in December and Flat Discount in
March. Before communicating this to the marketing team, the data scientist must first
verify that the difference is driven by genuine feature variation rather than model
noise. The investigation proceeds in four steps.

**Step 1 — Retrieve the feature vectors for both months**

Pull the two input rows passed to the model and display them side-by-side:

| Feature | Store 12, December | Store 12, March |
|---|:---:|:---:|
| `month` | 12 | 3 |
| `is_festival` | 1 | 0 |
| `festival_day_count` | 4 | 0 |
| `is_weekend` (proportion) | 0.36 | 0.29 |
| `competition_density` | 5 | 5 |
| `store_size` | medium | medium |
| `location_type` | semi-urban | semi-urban |
| `rolling_baseline` | 340 | 265 |

The structural features (store size, location, competition) are identical — confirming
the recommendation difference is driven entirely by temporal context. December has
festival days and a higher rolling baseline; March has neither.

**Step 2 — Score all five promotions for each month**

Run the model's predict step for both months with each of the five promotion types
substituted in turn. This produces a scoring matrix:

| Promotion | Dec score (predicted items_sold) | Mar score | Dec rank | Mar rank |
|---|:---:|:---:|:---:|:---:|
| Flat Discount | 318 | **288** | 3 | **1** |
| BOGO | 325 | 271 | 2 | 3 |
| Free Gift | 305 | 262 | 4 | 5 |
| Category Offer | 299 | 267 | 5 | 4 |
| **Loyalty Points** | **341** | 275 | **1** | 2 |

The table makes the recommendation transparent and auditable: it is not a black-box
"the model says so" but a ranked scorecard. The marketing team can see that Loyalty
Points wins December by 16 units over BOGO, while Flat Discount wins March by 13 units
over Loyalty Points — differences large enough to trust, not razor-thin margins.

**Step 3 — Attribute the difference using feature importance and partial dependence**

Use the Random Forest's feature importances to identify which features drove the
December/March difference. Two tools are most useful here:

- **Partial Dependence Plots (PDPs):** Show how predicted `items_sold` changes as
  `month` varies, holding all other features at Store 12's values. If Loyalty Points'
  PDP rises sharply in months 11–12 while Flat Discount's PDP peaks in months 2–4,
  this confirms the model has learned a genuine seasonal interaction: festive months
  reward loyalty mechanics (customers accumulate points as gifts); shoulder months
  reward price reductions when footfall is lower and price sensitivity is higher.

- **SHAP (SHapley Additive exPlanations) values** for the two predictions: a
  waterfall chart showing how each feature pushes the prediction above or below the
  baseline for each month. In December, `is_festival` and `month=12` should show
  large positive SHAP values for Loyalty Points. In March, `rolling_baseline` being
  lower and `is_festival=0` should show large positive SHAP values for Flat Discount.

**Step 4 — Communicate to the marketing team**

Translate the technical findings into business language using a one-page summary:

> *"In December, Store 12 benefits from four festival days and elevated footfall. Our
> model has learned from three years of data that during festival periods, customers
> respond more strongly to Loyalty Points — they are already motivated to buy, so a
> price cut provides little additional incentive, but accumulating reward points for
> the new year adds genuine perceived value. In March, footfall drops to its seasonal
> trough and no festivals occur. In that context, a Flat Discount is the stronger
> lever because price sensitivity is higher when customers are less motivated to visit."*

This narrative turns a model output into a hypothesis the marketing team can validate
against their own domain knowledge — and challenge if it conflicts with what they know
about their customers.


---

### *B3(c) — End-to-End Deployment and Monitoring* 

#### 1. Saving the trained model

The complete scikit-learn Pipeline (ColumnTransformer + regressor) is serialised as a
single binary artefact using `joblib`, which handles NumPy arrays and sparse matrices
more efficiently than Python's native `pickle`:

```python
import joblib

# Save the full pipeline — preprocessor + model in one object
joblib.dump(rf_pipeline, 'promotion_model_v1.2_20250101.joblib')

# Load at inference time — no retraining needed
pipeline = joblib.load('promotion_model_v1.2_20250101.joblib')
```

**Versioning discipline:** The filename encodes the model version and training
cut-off date. This is stored alongside a **model card** — a structured metadata
document recording: training date range, features used, hyperparameters, validation
RMSE/MAE/ranking accuracy, known limitations, and the name of the person who approved
it for production. Without a model card, a new team member cannot determine whether
the file on disk is current, deprecated, or experimental.

The artefact is committed to a **model registry** (e.g. MLflow Model Registry,
AWS SageMaker Model Registry) rather than a plain file share. A registry tracks
lineage (which training data and code version produced this model), stages
(Staging → Production → Archived), and enables one-click rollback to a prior version
if a newly deployed model degrades.

#### 2. Preparing and feeding new monthly data

At the start of each month, an automated pipeline performs the following steps:

**Step 1 — Extract:** Pull last month's completed transactions from the data warehouse,
join to `store_attributes`, `promotion_details`, and `calendar` using the same
star-schema join logic established at training time.

**Step 2 — Aggregate:** Apply the same store-month-promotion aggregations (SUM
`items_sold`, COUNT transactions, festival flags) and recompute the rolling 3-month
baseline using now-completed months. This is critical: the rolling baseline is a
**time-aware feature** that cannot be precomputed at training time for future months.

**Step 3 — Score all five promotions:** For each of the 50 stores, construct five
feature rows — one per promotion type — with the current month's calendar context,
last month's rolling baseline, and each store's fixed attributes. Feed all 250 rows
(50 stores × 5 promotions) through `pipeline.predict()` in a single vectorised call.

```python
# Build scoring dataframe: 50 stores × 5 promotions = 250 rows
scoring_df = build_scoring_matrix(stores, promotions, current_month_context)

# Score all combinations in one call — no loop needed
predictions = pipeline.predict(scoring_df)

# Reshape and find best promotion per store
scoring_df['predicted_items_sold'] = predictions
recommendations = (
    scoring_df
    .groupby('store_id')
    .apply(lambda g: g.loc[g['predicted_items_sold'].idxmax(), 'promotion_type'])
)
```

**Step 4 — Output:** Write the 50 recommendations to a `promotion_recommendations`
table in the data warehouse, timestamped with the run date and model version. The
marketing team reads from this table via their planning dashboard — they never
interact with the model directly.

**Step 5 — Retain actuals:** When the month ends, the actual `items_sold` for each
store-promotion combination is written to a `model_actuals` tracking table alongside
the model's prediction. This table is the foundation of the monitoring system.

#### 3. Monitoring for degradation and triggering retraining

**Metric tracking — continuous**

Every month, compute the following metrics over a rolling 3-month window using the
`model_actuals` table and alert if any breach a threshold:

| Metric | Alert threshold | What it signals |
|---|---|---|
| Rolling RMSE | > 1.5× baseline RMSE from validation | General prediction degradation |
| Rolling MAE | > 1.5× baseline MAE | Median error growing |
| Promotion ranking accuracy | < 50% (below 2.5× random) | Model failing its core task |
| Mean signed error (bias) | Abs value > 15 units sustained for 2+ months | Systematic drift — model consistently over or under-predicts |
| Feature distribution shift (PSI) | Population Stability Index > 0.2 for any key feature | Input data has changed structurally |

**Population Stability Index (PSI)** on the input feature distributions is the
*leading indicator* — it can detect distribution shift in the feature space *before*
prediction errors manifest. If `competition_density` or `festival_day_count`
distributions in live data diverge significantly from their training distributions,
the model is being asked to extrapolate outside its experience, and degradation will
follow even before it shows up in RMSE.

**Retraining triggers — semi-automatic**

Retraining is triggered when any of the following conditions are met:

1. **Metric alert:** RMSE or ranking accuracy breaches its threshold for two
   consecutive months (one-month spikes may be noise from an anomalous month).
2. **Scheduled retraining:** Regardless of metric alerts, the model is retrained
   every six months on a rolling window of the most recent 24 months. This ensures
   the model continuously incorporates evolving customer behaviour without requiring
   a crisis to prompt action.
3. **Structural business change:** A new store is opened, a store is reclassified to
   a different location type, a new promotion type is introduced, or major regulatory
   changes alter shopping behaviour. These are flagged by the business and trigger
   immediate retraining regardless of metric performance.

**Retraining process:** New training data is appended to the historical dataset (the
full window is not discarded — older data still provides valuable seasonal and
cross-store signal). The full pipeline — feature engineering, preprocessing, model
fitting, and hyperparameter validation — is re-executed via a CI/CD pipeline
(e.g. GitHub Actions triggering a SageMaker Training Job). The new model is
evaluated against the holdout from the most recent three months. If its ranking
accuracy exceeds the incumbent model's accuracy on the same holdout, it is promoted
to Production in the model registry. If not, the incumbent is retained and the data
science team investigates why the new training run underperformed before any further
action is taken.

This process ensures the model degrades gracefully, with human oversight at every
deployment gate, while running automatically enough to be sustainable for a 50-store
operation with monthly cadence.



```python

```
