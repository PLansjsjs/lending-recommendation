# Report Guide (Module 7 structure)

Map each report section to the files in `outputs/`. Run the real data first, then write from YOUR numbers.

| Report section | What to include | Source files |
|---|---|---|
| Summary | 5-6 lines: problem, data, best model, headline recommendation | `policy_evaluation.csv` |
| Problem identification | Lenders lose money on defaults but also lose good customers by rejecting them. One-size pricing ignores risk. | - |
| Objectives | (1) describe borrower/default patterns (2) model the interest rate (3) predict default (4) segment borrowers (5) recommend approve / adjust / reject | - |
| Data | Source, 50,000-row sample, target definition, variable types and scales, cleaning steps, missing values | `data_types_scales.csv`, `cleaning_log.csv`, `missing_values.csv` |
| Visualisation and exploration | Histograms, box plots, pie, bar charts by category, scatter, correlation heatmap | `02_*.png`, `descriptive_stats.csv` |
| Methodology | One paragraph per module: tests, regression, five classifiers, PCA/FA, K-means/hierarchical, expected-profit rule; train/test split, imputation inside pipelines | scripts |
| Interpretation: hypothesis tests | Which variables differ between defaulters and non-defaulters; **use effect sizes, because with ~40,000 rows nearly every p-value is tiny** | `hypothesis_tests.csv` |
| Interpretation: regression | Coefficients, R², adjusted R², RMSE, assumption checks (residuals, Q-Q, VIF, Breusch-Pagan) and what violations mean | `regression_*`, `mlr_summary.txt`, `03_*.png` |
| Interpretation: classification | Comparison table, ROC curves, confusion matrices, why accuracy misleads at ~20% defaults, odds ratios, tree rules | `classifier_comparison.csv`, `04_*.png`, `logit_odds_ratios.csv`, `tree_rules.txt` |
| Interpretation: clustering | PCA variance, factor loadings, chosen k and why, segment names and traits | `05_*.png`, `cluster_profile.csv`, `pca_loadings.csv` |
| Findings / recommendations | Action per segment, share approved / adjusted / rejected, profit vs baseline, sensitivity | `segment_recommendations.csv`, `sensitivity_analysis.csv`, `06_*.png` |
| Conclusions and limitations | See below | - |

## Points that make the report credible

* **Data leakage:** you excluded `grade`, `int_rate` and post-loan payment columns from the default model. Say why.
* **Imbalance:** report recall, AUC and the Youden-threshold rows, not only accuracy.
* **Honest evaluation:** compare **Policy A (reject only)** to the baseline. Policy B assumes every borrower
  accepts a higher rate, so treat its profit as an optimistic upper bound.
* **Assumptions:** LGD and the target margin are assumed, not measured. Show the sensitivity table.
* **Selection bias:** the data contains only loans that were approved and finished. Rejected applicants and
  still-running loans are missing, so the model may not generalise to everyone who applies.
* **Time:** default patterns change with the economy and with issue year (see the default-by-year chart).
* **Fairness:** automated lending decisions need human review and checks for unfair impact on groups.
  This is a classroom model, not a real credit policy.
* **Low AUC is normal:** default prediction without the lender's own grade is hard; around 0.65 to 0.72 is typical.
  Explain it rather than hiding it.
