# Customer Intelligence & Support Copilot

## Overview

A reproducible, offline Python project using the supplied synthetic Instamart-style data. It contains a data-quality pipeline and SQLite warehouse (Part A), customer segmentation and lapse prediction (Part B), privacy-safe support-ticket triage (Part C), and a TF-IDF policy assistant plus tool-using refund agent (Part D). All model calls use the supplied `mock_llm.py`; there are no API keys, hosted models, network calls or paid services. All amounts are INR.

The supplied `generate_data.py` and `mock_llm.py` files are included unchanged. The scripts under `src/` implement the project pipeline.

## Reproduce

Run these commands from the repository root using Python 3.11 or newer. On Windows, activate the environment with `.venv\\Scripts\\activate`; on macOS/Linux use `source .venv/bin/activate`.

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

`generate_data.py` must print `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The pipeline then prints `ALL CHECKPOINTS PASSED` after Part A validation. The scripts create the required `data/clean/`, `data/instamart.db`, `outputs/` and `state/` artifacts. The generated synthetic data is deterministic.

## Part A

### Audit and cleaning

Raw table shapes: customers **(1,200, 5)**, products **(40, 4)**, orders **(13,084, 9)**. Raw inferred dtypes: customers (`customer_id`, `city`, `signup_date`, `age`, `gender`) are object; products (`product_id`, `product_name`, `category`) are object and `price_inr` is int64; orders (`order_id`, `customer_id`, `product_id`, `order_ts`, `status`, `order_amount`, `delivery_minutes`) are object and `quantity`, `discount_pct` are int64. Per-column missing counts and percentages are written to `outputs/audit_missingness.csv`.

| Raw audit item | Result |
|---|---:|
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Unparseable order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

Customer cleaning: trim and title-case city; map gender variants to `male`, `female` or `unknown`; mark ages outside 10–100 missing and fill missing ages with the city median using `groupby().transform()`; parse signup dates. Clean city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. No ages are missing; all are between 18 and 60.

Order cleaning is applied in this order: parse `order_ts`; remove exact duplicate rows; retain the latest timestamp per `order_id` to resolve pending/final versions; coerce amount to numeric and impute missing values using catalogue price × quantity × (1 − discount/100); set negative delivery durations missing; impute missing durations only for completed orders using the completed-order median of the product category. Cancelled-order delivery times remain missing because no delivery occurred. Clean orders: 12,309 rows, 11,110 completed and 1,199 cancelled; no pending rows; no missing amounts; zero missing delivery times among completed orders and 1,199 among cancelled orders. Completed-order revenue is **INR 2,998,668.80**.

The clean CSVs and SQLite tables (`customers`, `products`, `orders`) are created in `data/clean/` and `data/instamart.db`. The validation script asserts the key cleaning checkpoints and prints the required success line.

### SQL and pandas analysis

Seven runnable SQL statements are in `sql/queries.sql`; each is executed through `pandas.read_sql_query` and saved as `outputs/sql_q1.csv` through `outputs/sql_q7.csv`.

| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 city × category revenue join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 products above average catalogue price | 11 products |
| Q4 customers with no completed order | 25 customers |
| Q5 CTE: customers above average completed spend | 550 customers; average spend INR 2,552.06 |
| Q6 ranking windows | Top customers: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 daily running revenue | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Ties share a rank in `RANK()` but still receive distinct sequential positions in `ROW_NUMBER()`, which is why the two columns differ for eight rows. The pandas city/category pivot table reproduces the SQL grand total. City aggregates (sum / mean / count) include Bengaluru **523,643.40 / 276.47 / 1,894** and Mumbai **479,058.50 / 297.00 / 1,613**.

Completed-order amount statistics: mean **269.91**, median **135.00**, skewness **4.61**, Q1 **64.80**, Q3 **297.00**. The 1.5×IQR fences are approximately **−283.50** and **645.30**; there are **0** lower-fence outliers and **897** upper-fence outliers. The distribution is strongly right-skewed: most orders are relatively small, while a smaller set of expensive orders pulls the mean above the median. The correlation with the largest absolute magnitude is order amount versus discount percentage (approximately **−0.095**); all four measured correlations are weak. The data represents synthetic customer orders, products, delivery performance and spending across six cities; no strong linear relationship is evident among amount, discount, quantity and delivery time.

## Part B

### RFM and segments

The snapshot is 2025-10-01. The RFM table contains **1,175** customers; recency median/min/max = **19 / 0 / 180** days; frequency mean/max = **9.46 / 33**; monetary mean/max = **INR 2,552.06 / INR 14,766.55**. K-means uses `StandardScaler` and `KMeans(n_clusters=K, n_init=10, random_state=42)` for K=2…8.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: it has the strongest silhouette in this sweep and provides four interpretable campaign groups. Inertia necessarily falls as K increases because more centroids can fit the observations more closely; K=8's lower inertia alone is not evidence that eight useful business segments exist, so silhouette and interpretability also matter.

| Segment ID | Business name | Size | Mean recency | Mean frequency | Mean monetary (INR) | Action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Use low-friction reminders and replenishment offers. |
| 1 | Premium Champions | 138 | 24.12 | 6.77 | 7,129.70 | Protect with priority support and relevant premium bundles. |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | 3,410.02 | Encourage repeat baskets and subscriptions. |
| 3 | Occasional Shoppers (Dormant) | 182 | 109.04 | 2.20 | 685.99 | Run a win-back campaign with a time-limited offer. |

Segment 1 is marked Premium because it has the highest mean monetary value. Cluster IDs are arbitrary labels, not an ordinal scale. PCA explained-variance ratios are **0.6343** and **0.2180** (total **0.8523**). Results are in `outputs/customer_segments.csv`, `outputs/segment_profiles.csv`, `outputs/kmeans_metrics.csv` and `outputs/pca_variance.csv`.

### Lapse model

The pre-cutoff cohort (cutoff 2025-09-01) contains **1,156** customers; **443 (38.32%)** lapsed. The stratified split is **924 train / 232 test**. A constant not-lapsed baseline gives accuracy **61.6%** and lapsed-class F1 **0.00**. Features use only completed/final-status orders before 2025-09-01 plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite category, city, age, gender and tenure. No data dated 2025-09-01 or later entered model features, and Part B segment labels were not features. All encoders, scalers and resampling were fitted on training rows only; test rows were only transformed and used for final evaluation. A code assertion checks that no feature column contains missing values.

Grid search: Logistic Regression `C ∈ {0.01, 0.1, 1, 10, 100}`, 5-fold F1; best `C=100`, CV F1 **0.651**. Random Forest `max_depth ∈ {3,5,8}`, `min_samples_leaf ∈ {5,10,20}`, 100 trees, 5-fold F1; best depth **8**, leaf **10**, CV F1 **0.681**.

| Model | Confusion matrix (TN, FP / FN, TP) | Precision (lapsed) | Recall (lapsed) | F1 (lapsed) |
|---|---|---:|---:|---:|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

Random Forest is selected (test F1 **0.655**, above the 0.55 acceptance threshold). Missing a customer who is about to lapse loses a retention opportunity and potentially future revenue. Incorrectly targeting a loyal customer wastes an incentive and can train customers to expect discounts. The forest has slightly higher precision with the same recall in this split, so it is the more conservative choice while still identifying a useful share of lapsing customers.

Training-only oversampling duplicates minority-class training examples to reach **570 / 570**. Balanced logistic regression gives precision **0.589**, recall **0.854**, F1 **0.697**, compared with unbalanced logistic precision **0.644**, recall **0.652**, F1 **0.648**. Recall moved most; the retention team catches more potential lapsers, at the cost of more false-positive campaigns.

The five largest Random Forest importances are frequency (0.280), monetary (0.271), recency (0.115), average delivery (0.061) and cancellation rate (0.052). The largest-magnitude logistic coefficients are frequency (negative), favourite category Electronics (negative) and favourite category Grocery (negative); holding other features constant, these negative coefficients reduce the model's estimated lapse odds relative to the corresponding reference levels. The fitted training scaler's frequency mean is **8.115**, versus full-cohort mean **8.029**, because the scaler sees only the training partition. The selected model is saved and reloaded from `outputs/lapse_model.joblib`; the script asserts exact equality of predictions after reload.

`outputs/retention_list.csv` contains the top 10 Premium test customers by predicted lapse probability; **1** truly lapsed in this list. The Premium flag is a post-hoc whole-window label based on RFM through September, so the retention list is illustrative and is not a leakage-free evaluation.

## Part C

Tickets are processed in file order. Four empty tickets are marked `skipped_empty`; four long tickets (734 estimated tokens each) are marked `flagged_long`; neither group is sent to the model. Email, phone and order-ID redaction is applied in the required order and rechecked: **55 emails + 91 phone numbers + 132 order IDs = 278 substitutions**, with zero remaining matches. The model's user message contains only the redacted ticket text.

The zero-shot and few-shot prompts both specify CLASSIFY, the four allowed categories and JSON-only output. The few-shot prompt includes `EXAMPLES:` and labelled billing-independent examples including fraud. Every model call is retried on `TransientLLMError` up to three retries using the required exponential jitter; the full run recovers all **19** transient failures.

| Prompt | Classified tickets | Accuracy on classified | Total tokens | Tokens / classified |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 9,088 | 61.82 |
| Few-shot | 147 | 100.0% (147/147) | 13,144 | 89.41 |

Zero-shot confusion table:

| True category | Predicted billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about **27.59 additional tokens per classified ticket** and increases total token use by about 44.6%, but it corrects all 21 zero-shot errors (11 fraud and 10 technical tickets incorrectly labelled billing). That extra cost is justified when category accuracy and correct routing are important, especially for fraud and security cases; a lower-cost zero-shot mode may be suitable for low-risk triage with human review.

The 156 eligible tickets yielded **135 `classified_json`, 12 `classified_keyword`, and 9 `flagged_invalid_output`**. The invalid-output tickets, empty tickets, long tickets and budget-stopped tickets route to human review; accepted categories are rule-validated before routing. With TOKEN_BUDGET **6,000**, **53** tickets ended as `budget_exhausted`. The final few-shot run uses budget 100,000 and creates `outputs/triage_results.csv` with 164 rows. Fraud routes to the high-priority security team; otherwise Premium customers go to the high-priority premium desk; remaining valid tickets go to the support queue. `outputs/triage_budget_demo.csv`, `outputs/triage_confusion_zero.csv`, `outputs/triage_comparison.json` and `outputs/redaction_counts.json` preserve the evidence.

## Part D

### Policy RAG

The five policy files produce **15 chunks** using 40-word chunks and 10-word overlap: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on chunk text; retrieval ranks cosine similarity and returns source, score and text. Term frequency measures how often a term appears in a chunk, while inverse document frequency downweights terms common across many chunks. TF-IDF is lexical, so semantically equivalent phrasings that use different words can retrieve poorly.

With overlap, all **12/12** answerable questions retrieve the expected document and all **12/12** answers contain the expected fact; both unanswerable questions receive the required fallback (**2/2**). With zero overlap, **10/12** answerable questions still contain the expected fact (the policy document is still retrieved for both), while both unanswerable questions fall back. The questions about the completed-order late-delivery refund threshold and the 48-hour failed-payment return are affected by chunk-boundary context loss in the zero-overlap split. Results are saved in `outputs/rag_evaluation.csv` and `outputs/rag_zero_overlap.csv`.

### Refund agent

The tool map implements `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. The agent uses a ReAct loop with a maximum of six iterations, a 6,000-token limit and a 10-second time limit. All seven scenarios:

| Scenario | Status | Refund amount | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 892 |
| S2 | REFUND | INR 135.00 | 5 | 892 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 879 |
| S4 | DENIED | INR 0.00 | 4 | 601 |
| S5 | DENIED | INR 0.00 | 3 | 334 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | 2,994 |

The S1 run with token limit 200 ends `FALLBACK_TOKEN_LIMIT`; the zero-second demonstration ends `FALLBACK_TIMEOUT`. S7 completes all six model calls and ends with `FALLBACK_MAX_ITERATIONS`. The crash-and-resume S1 demonstration saved a checkpoint after its second tool observation, resumed without repeating work, and finished with REFUND INR 128.25. Tools executed across the crash and resume, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. Checkpoints are in `state/`; run details are in `outputs/agent_runs.jsonl`; refund actions are in `outputs/refund_ledger.jsonl`.

ReAct uses a model call after each observation (five calls for S1, three for S5 and two for S6), so it incurs repeated prompt overhead but can react to actual order results. A plan-and-execute approach could request a complete action plan in one call and use fewer tokens for normal cases, but a NOT_FOUND order or cancelled order could invalidate the rest of that plan. For refunds, I prefer ReAct because the decision to refund, deny or escalate depends on verified database observations and policy retrieval; its extra calls buy safer branching and auditable tool steps.

## Limitations and Next Steps

The data and policies are synthetic and the offline mock model is deterministic; results should not be interpreted as real customer or operational performance. The classifier and refund rules are intentionally narrow, and production deployment would require access controls, monitoring, human review, secure audit retention and broader policy tests. The Premium flag is a post-hoc whole-window label derived from RFM through September, so the retention list is illustrative and not a leakage-free evaluation. Future work should evaluate a time-consistent Premium definition, calibrate lapse probabilities, monitor drift and test the pipeline on appropriately governed real-world data.
