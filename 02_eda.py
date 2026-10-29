"""Step 2 (Module 2) - descriptive analytics, visualisation, hypothesis tests."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from common import load_clean, save_fig, save_table

NUMERIC = ["loan_amnt", "int_rate", "annual_inc", "dti", "revol_util", "revol_bal", "open_acc",
           "total_acc", "emp_length", "delinq_2yrs", "inq_last_6mths", "pub_rec", "loan_to_income"]
CATEGORICAL = ["term", "purpose", "home_ownership", "verification_status", "grade"]


def descriptive(df):
    d = df[NUMERIC].agg(["count", "mean", "median", "std", "min", "max", "skew", "kurt"]).T
    d["iqr"] = df[NUMERIC].quantile(0.75) - df[NUMERIC].quantile(0.25)
    d["cv"] = d["std"] / d["mean"]
    d["missing_%"] = df[NUMERIC].isna().mean() * 100
    save_table(d.round(3), "descriptive_stats.csv")

    overall = df["default"].mean()
    lo, hi = proportion_confint(df["default"].sum(), len(df), method="wilson")
    print(f"  Overall default probability = {overall:.4f}  (95% CI {lo:.4f} to {hi:.4f})")
    by_term = df.groupby("term")["default"].agg(["mean", "count"])
    print("  P(default | term):\n", by_term.round(4).to_string())


def figures(df):
    # histograms
    cols = [("loan_amnt", "Loan amount ($)"), ("log_income", "log(Annual income)"),
            ("dti", "Debt-to-income (%)"), ("int_rate", "Interest rate (%)"),
            ("revol_util", "Revolving utilisation (%)"), ("emp_length", "Employment length (yrs)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (c, label) in zip(axes.ravel(), cols):
        ax.hist(df[c].dropna(), bins=40, color="#3b6ea5", edgecolor="white")
        ax.set_title(label)
    save_fig("02_histograms.png")

    # box plots by outcome
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, (c, label) in zip(axes, [("dti", "DTI"), ("log_income", "log(Income)"),
                                    ("int_rate", "Interest rate"), ("loan_amnt", "Loan amount")]):
        groups = [df.loc[df["default"] == k, c].dropna() for k in (0, 1)]
        ax.boxplot(groups, showfliers=False)
        ax.set_xticklabels(["Paid", "Default"])
        ax.set_title(label)
    save_fig("02_boxplots_by_default.png")

    # default rate by category (bar charts)
    cats = [c for c in CATEGORICAL if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), cats):
        rate = df.groupby(c)["default"].mean().mul(100).sort_values()
        ax.barh(rate.index.astype(str), rate.values, color="#c0504d")
        ax.set_title(f"Default rate (%) by {c}")
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    save_fig("02_default_rate_by_category.png")

    # pie chart of the outcome
    plt.figure(figsize=(4.5, 4.5))
    counts = df["default"].value_counts().sort_index()
    plt.pie(counts, labels=["Fully Paid", "Charged Off"], autopct="%1.1f%%",
            colors=["#6aa84f", "#c0504d"], startangle=90)
    plt.title("Loan outcome share")
    save_fig("02_pie_outcome.png")

    # scatter + trend line
    s = df[["dti", "int_rate"]].dropna().sample(min(5000, len(df)), random_state=1)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(s["dti"], s["int_rate"], s=6, alpha=0.3)
    slope, intercept = np.polyfit(s["dti"], s["int_rate"], 1)
    xs = np.linspace(s["dti"].min(), s["dti"].max(), 50)
    plt.plot(xs, intercept + slope * xs, color="red")
    plt.xlabel("DTI (%)"); plt.ylabel("Interest rate (%)"); plt.title("DTI vs interest rate")
    save_fig("02_scatter_dti_rate.png")

    # correlation heatmap
    cols = [c for c in NUMERIC if c != "revol_bal"] + ["default"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=75, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Correlation matrix")
    save_fig("02_correlation_heatmap.png")

    # normality check (Q-Q plot)
    plt.figure(figsize=(5, 4.5))
    stats.probplot(df["log_income"].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q plot: log(income)")
    save_fig("02_qq_log_income.png")

    if "issue_year" in df and df["issue_year"].notna().any():
        yr = df.groupby("issue_year")["default"].mean().mul(100)
        plt.figure(figsize=(6, 4))
        plt.plot(yr.index, yr.values, marker="o")
        plt.xlabel("Issue year"); plt.ylabel("Default rate (%)"); plt.title("Default rate by issue year")
        save_fig("02_default_rate_by_year.png")


def hypothesis_tests(df):
    rows = []
    d1, d0 = df[df["default"] == 1], df[df["default"] == 0]
    for c in ["annual_inc", "loan_amnt", "dti", "revol_util", "int_rate", "emp_length",
              "loan_to_income", "inq_last_6mths", "delinq_2yrs"]:
        a, b = d1[c].dropna(), d0[c].dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        p_mw = stats.mannwhitneyu(a, b).pvalue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"test": "Welch t-test (defaulted vs paid)", "variable": c,
                     "mean_default": a.mean(), "mean_paid": b.mean(), "statistic": t, "p_value": p,
                     "mann_whitney_p": p_mw, "effect_size": (a.mean() - b.mean()) / pooled,
                     "effect_size_type": "Cohen d"})
    for c in [c for c in ["term", "purpose", "home_ownership", "verification_status", "grade"] if c in df]:
        table = pd.crosstab(df[c], df["default"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        rows.append({"test": "Chi-square independence", "variable": c, "statistic": chi2,
                     "p_value": p, "effect_size": v, "effect_size_type": "Cramer V"})
    # two-proportion z-test: 60-month vs 36-month loans
    g60, g36 = df[df["term"] == 60], df[df["term"] == 36]
    if len(g60) and len(g36):
        z, p = proportions_ztest([g60["default"].sum(), g36["default"].sum()], [len(g60), len(g36)])
        rows.append({"test": "Two-proportion z-test (60m vs 36m)", "variable": "term",
                     "mean_default": g60["default"].mean(), "mean_paid": g36["default"].mean(),
                     "statistic": z, "p_value": p})
    res = pd.DataFrame(rows)
    res["reject_H0_at_5%"] = res["p_value"] < 0.05
    save_table(res.round(5), "hypothesis_tests.csv", index=False)

    # correlation of each numeric variable with default
    corr_rows = []
    for c in NUMERIC:
        m = df[[c, "default"]].dropna()
        r, p = stats.pearsonr(m[c], m["default"])
        rho, p_s = stats.spearmanr(m[c], m["default"])
        corr_rows.append({"variable": c, "pearson_r": r, "pearson_p": p, "spearman_rho": rho, "spearman_p": p_s})
    save_table(pd.DataFrame(corr_rows).round(5).sort_values("pearson_r", key=abs, ascending=False),
               "correlation_with_default.csv", index=False)

    stat, p = stats.normaltest(df["log_income"].dropna().sample(min(5000, len(df)), random_state=1))
    print(f"  Normality test on log(income): p = {p:.4g}, skewness = {df['log_income'].skew():.2f}")
    print("\n  NOTE: with tens of thousands of rows almost every p-value is tiny. "
          "Discuss the effect_size column, not just p < 0.05.")


def main():
    df = load_clean()
    print("Descriptive statistics"); descriptive(df)
    print("Figures"); figures(df)
    print("Hypothesis tests"); hypothesis_tests(df)


if __name__ == "__main__":
    main()
