"""Recommendation engine: turns model outputs into Approve / Adjust rate / Reject.

For every borrower it combines
  * probability of default  (best classifier, step 4)
  * benchmark interest rate (multiple regression, step 3)
  * risk segment            (PCA + K-means, step 5)
and applies an expected-profit rule using the assumptions in common.py.
"""
import joblib
import numpy as np
import pandas as pd
from common import (LGD, MAX_RATE, MIN_RATE, MODEL_DIR, RATE_STEP, TARGET_MARGIN, add_derived,
                    reg_matrix)

RATE_GRID = np.arange(MIN_RATE, MAX_RATE + 1e-9, RATE_STEP)


def load_models():
    needed = ["best_classifier.joblib", "regression_model.joblib", "cluster_model.joblib"]
    missing = [f for f in needed if not (MODEL_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing model files {missing}. Run steps 03, 04 and 05 first.")
    return tuple(joblib.load(MODEL_DIR / f) for f in needed)


def interest_per_dollar(rate, months):
    """Total interest paid on a fully amortising loan, per $1 of principal."""
    i = np.asarray(rate, float) / 12
    n = np.asarray(months, float)
    return n * i / (1 - (1 + i) ** (-n)) - 1


def expected_profit(pd_, rate, months, lgd=LGD):
    """Expected profit per $1 lent over the life of the loan."""
    return (1 - pd_) * interest_per_dollar(rate, months) - pd_ * lgd


def realized_profit(default, rate, months, lgd=LGD):
    """Profit per $1 once the outcome is known (default loses `lgd`, otherwise earns the interest)."""
    return np.where(np.asarray(default) == 1, -lgd, interest_per_dollar(rate, months))


def apply_policy(p_default, benchmark_rate, months, lgd=LGD, target=TARGET_MARGIN):
    """Decision rule.

    1. If the loan already meets the target margin at the benchmark rate -> APPROVE
    2. Else find the lowest rate (up to MAX_RATE) that meets it            -> ADJUST RATE
    3. Else no allowed rate is profitable enough                           -> REJECT
    """
    p = np.asarray(p_default, float)
    bench = np.clip(np.asarray(benchmark_rate, float), MIN_RATE, MAX_RATE)
    m = np.asarray(months, float)
    ep_bench = expected_profit(p, bench, m, lgd)

    grid = RATE_GRID[None, :]
    ok = (expected_profit(p[:, None], grid, m[:, None], lgd) >= target) & (grid >= bench[:, None])
    can_fix = ok.any(axis=1)
    required = np.where(can_fix, RATE_GRID[ok.argmax(axis=1)], np.nan)

    meets = ep_bench >= target
    action = np.where(meets, "APPROVE", np.where(can_fix, "ADJUST RATE", "REJECT"))
    rate = np.where(meets, bench, required)
    with np.errstate(invalid="ignore"):
        ep_final = np.where(action == "REJECT", np.nan, expected_profit(p, rate, m, lgd))
    return {"action": action, "recommended_rate": rate, "required_rate": required,
            "expected_profit": ep_final}


def predict_benchmark_rate(df, reg):
    X = reg_matrix(df, reg["medians"], reg["full_columns"])[reg["cols"]]
    rate = reg["params"]["const"] + X.values @ reg["params"][reg["cols"]].values
    return np.clip(rate / 100, MIN_RATE, MAX_RATE)


def assign_segment(df, clu):
    X = clu["imputer"].transform(df[clu["features"]])
    labels = clu["kmeans"].predict(clu["pca"].transform(clu["scaler"].transform(X)))
    return pd.Series(labels).map(clu["segment_names"]).values


def score_borrowers(df, models=None):
    """Model outputs only (no policy): default probability, benchmark rate, segment."""
    clf, reg, clu = models or load_models()
    df = add_derived(df)
    out = df.copy()
    out["p_default"] = clf["pipeline"].predict_proba(df[clf["features"]])[:, 1]
    out["benchmark_rate"] = predict_benchmark_rate(df, reg)
    out["segment"] = assign_segment(df, clu)
    return out


def recommend_df(df, models=None, lgd=LGD, target=TARGET_MARGIN):
    """Full recommendation for a DataFrame of borrowers."""
    out = score_borrowers(df, models)
    pol = apply_policy(out["p_default"], out["benchmark_rate"], out["term"], lgd, target)
    out["action"] = pol["action"]
    out["recommended_rate"] = pol["recommended_rate"]
    out["required_rate"] = pol["required_rate"]
    out["expected_profit"] = pol["expected_profit"]
    return out
