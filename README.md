# Lending Club Loan Recommendation Model (BMGT107L DA)

**Problem:** A lender must decide, for each loan applicant, whether to **approve**, **approve at a
higher (adjusted) interest rate**, or **reject**. The model predicts default risk, estimates a fair
benchmark interest rate, groups borrowers into risk segments, and then recommends an action per
borrower and per segment using an expected-profit rule.

No Jupyter needed: everything is plain `.py` files you run from VS Code or the terminal.
No GPU needed either (all techniques are classical scikit-learn / statsmodels).

---

## 1. One-time setup (Windows, PowerShell, Python 3.10)

Open this folder in VS Code (**File > Open Folder > lending_da**), then **Terminal > New Terminal**:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

* If PowerShell blocks the activate script, run once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
* If you already have a venv for another project, you can reuse it: just run
  `pip install -r requirements.txt` inside it.
* In VS Code press `Ctrl+Shift+P` > **Python: Select Interpreter** > pick the `venv` one,
  so the green Run button (top right of any file) uses it.

## 2. Try it first on fake data (2 minutes)

```powershell
python run_all.py --demo
```

This creates a **synthetic** file, runs all 7 steps and fills the `outputs/` folder. It proves your
setup works. **Never report numbers from the demo.** Delete `data\demo_loan_synthetic.csv` afterwards.

## 3. Use the real data

1. On Kaggle, search **"Lending Club"** and download a loan dataset (common versions are
   `accepted_2007_to_2018Q4.csv(.gz)` or `loan.csv`). Check the licence on the dataset page.
2. Copy the file into the `data\` folder (a `.csv.gz` works without unzipping).
3. Run:

```powershell
python run_all.py
```

Step 0 reads only the columns needed, in chunks (so it will not crash your laptop) and saves a
50,000-row random sample. If your file uses different column names, step 0 tells you which are
missing; edit `RAW_COLUMNS` in `common.py`.

I could not download Kaggle files in my environment, so the code was tested on synthetic data in
the same format. Expect to maybe tweak one small thing for the real file. Results (AUC, R², segments)
will differ from the demo.

You can also run any step on its own with the green Run button, in this order.

## 4. What each step does and produces

| Step | Script | Syllabus module | Main outputs (in `outputs/`) |
|---|---|---|---|
| 0 | `00_make_sample.py` | M1 | `data/loan_sample.csv` |
| 1 | `01_clean.py` | M1 data types, scales, cleansing | cleaning_log, missing_values, data_types_scales |
| 2 | `02_eda.py` | M2 descriptive stats, plots, hypothesis tests, p-values | histograms, box plots, pie, scatter, heatmap, Q-Q; hypothesis_tests.csv |
| 3 | `03_regression.py` | M3 simple + multiple regression, assumptions, model comparison | regression_model_comparison, mlr_summary.txt, diagnostics, VIF |
| 4 | `04_classification.py` | M4 logistic, tree, KNN, Naive Bayes, LDA, ROC, confusion matrix | classifier_comparison, ROC curves, confusion matrices, odds ratios, tree rules |
| 5 | `05_clustering.py` | M5 PCA, factor analysis, K-means, hierarchical | scree, loadings, elbow/silhouette, dendrogram, cluster_profile |
| 6 | `06_recommend.py` | M6 financial analytics | segment_recommendations, policy_evaluation, sensitivity_analysis, recommendations_test_set |
| - | `demo_borrower.py` | the "software" demo | prints a recommendation for individual applicants |

`recommender.py` holds the decision logic; `common.py` holds all settings.

## 5. How the recommendation works

1. **Default probability** from the best classifier (chosen by cross-validated AUC).
2. **Benchmark rate** from the multiple regression (what similar borrowers normally pay).
3. **Risk segment** from PCA + K-means.
4. **Expected profit per $1 lent** = (1 - p) x interest earned - p x loss given default.
5. Rule: **APPROVE** if the benchmark rate already meets the target margin; **ADJUST RATE** if a
   higher rate (up to the cap) does; otherwise **REJECT**.

To demo it live, edit the applicants in `demo_borrower.py` and run it.

## 6. Settings you can change (`common.py`)

* `LGD` (60%), `TARGET_MARGIN` (3%), `MIN_RATE`, `MAX_RATE`: **assumptions**, not facts from the data.
  `06_recommend.py` prints a sensitivity table over them; include it in your report.
* `SAMPLE_SIZE`, `SEED`, `TEST_SIZE`.
* `USE_LC_RISK_FEATURES`: leave `False`. `grade` and `int_rate` are Lending Club's own risk
  assessment, so using them to predict default is circular (data leakage).

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The venv is not active or the interpreter is wrong: activate it and `pip install -r requirements.txt` |
| Step 0 is slow or memory heavy | Normal for the full file; it reads in chunks. Close other apps. |
| "columns are missing" | Edit `RAW_COLUMNS` in `common.py` to match your file's headers |
| Figures do not pop up | By design, they are saved to `outputs\figures\` |
| Want a fresh run | Delete `data\loan_sample.csv`, `data\loan_clean.csv` and the `outputs\` sub-folders contents |
