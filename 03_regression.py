"""Step 3 (Module 3) - simple and multiple linear regression on the interest rate.

Question: what interest rate do borrowers with a given profile normally pay?
The best model becomes the "benchmark rate" used by the recommender (step 6).
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from common import (MODEL_DIR, REG_BASE, REG_CREDIT, TAB_DIR, get_split, reg_matrix,
                    save_fig, save_table)


def main():
    train, test = get_split()
    num_cols = REG_BASE + REG_CREDIT
    medians = train[num_cols].median()           # learned from training data only
    Xtr = reg_matrix(train, medians)
    full_columns = list(Xtr.columns)
    Xte = reg_matrix(test, medians, full_columns)
    ytr, yte = train["int_rate"], test["int_rate"]

    specs = {
        "SLR-1: term": ["term"],
        "SLR-2: dti": ["dti"],
        "MLR-A: dti, log_income, loan_amnt, term": REG_BASE,
        "MLR-B: A + credit history": REG_BASE + REG_CREDIT,
        "MLR-C: B + categorical variables": full_columns,
    }

    fitted, rows = {}, []
    for name, cols in specs.items():
        m = sm.OLS(ytr, sm.add_constant(Xtr[cols], has_constant="add")).fit()
        pred = m.predict(sm.add_constant(Xte[cols], has_constant="add"))
        err = yte - pred
        fitted[name] = (m, cols)
        rows.append({
            "model": name, "n_predictors": len(cols), "R2": m.rsquared, "adj_R2": m.rsquared_adj,
            "AIC": m.aic, "BIC": m.bic, "F_test_p": m.f_pvalue,
            "test_RMSE": np.sqrt((err ** 2).mean()), "test_MAE": err.abs().mean(),
            "test_R2": 1 - (err ** 2).sum() / ((yte - yte.mean()) ** 2).sum(),
        })
    comp = pd.DataFrame(rows)
    save_table(comp.round(4), "regression_model_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))

    # best multiple regression = lowest test RMSE among the MLR models
    mlr = comp[comp["model"].str.startswith("MLR")]
    best_name = mlr.loc[mlr["test_RMSE"].idxmin(), "model"]
    best, best_cols = fitted[best_name]
    print(f"\nBest MLR model: {best_name}")

    (TAB_DIR / "mlr_summary.txt").write_text(best.summary().as_text(), encoding="utf-8")
    coef = pd.DataFrame({"coef": best.params, "std_err": best.bse, "t": best.tvalues,
                         "p_value": best.pvalues})
    coef[["ci_low", "ci_high"]] = best.conf_int()
    save_table(coef.round(5), "mlr_coefficients.csv")

    # ---- assumption checks
    resid, fit = best.resid, best.fittedvalues
    idx = np.random.default_rng(1).choice(len(resid), size=min(5000, len(resid)), replace=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].scatter(fit.iloc[idx], resid.iloc[idx], s=5, alpha=0.3)
    axes[0].axhline(0, color="red"); axes[0].set_title("Residuals vs fitted (linearity, constant variance)")
    axes[0].set_xlabel("Fitted"); axes[0].set_ylabel("Residual")
    stats.probplot(resid, dist="norm", plot=axes[1]); axes[1].set_title("Q-Q plot of residuals (normality)")
    axes[2].hist(resid, bins=50, color="#3b6ea5", edgecolor="white"); axes[2].set_title("Residual histogram")
    save_fig("03_regression_diagnostics.png")

    exog = best.model.exog[:10000]
    with np.errstate(all="ignore"):
        vif = pd.DataFrame({"variable": best.model.exog_names[1:],
                            "VIF": [variance_inflation_factor(exog, i) for i in range(1, exog.shape[1])]})
    save_table(vif.round(3), "regression_vif.csv", index=False)
    lm, lm_p, f, f_p = het_breuschpagan(resid, best.model.exog)
    print(f"Max VIF = {vif['VIF'].max():.2f} (above ~5-10 signals multicollinearity)")
    print(f"Breusch-Pagan test for constant variance: p = {lm_p:.4g} (p < 0.05 means heteroscedasticity)")

    # predicted vs actual on the test set
    pred = best.predict(sm.add_constant(Xte[best_cols], has_constant="add"))
    plt.figure(figsize=(5.5, 5))
    plt.scatter(yte.iloc[:4000], pred.iloc[:4000], s=5, alpha=0.3)
    lim = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
    plt.plot(lim, lim, color="red"); plt.xlabel("Actual rate (%)"); plt.ylabel("Predicted rate (%)")
    plt.title("Test set: predicted vs actual interest rate")
    save_fig("03_regression_pred_vs_actual.png")

    joblib.dump({"model_name": best_name, "cols": best_cols, "full_columns": full_columns,
                 "params": best.params, "medians": medians}, MODEL_DIR / "regression_model.joblib")
    print("  saved model : outputs/models/regression_model.joblib")


if __name__ == "__main__":
    main()
