"""Step 6 - the recommendation model.

Applies the policy to the HELD-OUT test borrowers, then evaluates it against what
really happened (defaults) and shows how sensitive it is to the assumptions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import LGD, TARGET_MARGIN, TAB_DIR, get_split, save_fig, save_table
from recommender import (apply_policy, interest_per_dollar, load_models, realized_profit,
                         recommend_df)

ACTIONS = ["APPROVE", "ADJUST RATE", "REJECT"]
COLORS = {"APPROVE": "#6aa84f", "ADJUST RATE": "#f1c232", "REJECT": "#c0504d"}


def scenarios(test, action, rec_rate, lgd):
    """Realised profit under three strategies, using the real outcomes."""
    w = test["loan_amnt"].values
    d, months = test["default"].values, test["term"].values
    actual = test["int_rate"].values / 100
    funded_screen = action != "REJECT"
    repriced = np.where(action == "ADJUST RATE", np.maximum(actual, rec_rate), actual)
    out = {}
    for name, funded, rate in [
        ("Baseline: lend to everyone at the actual rate", np.ones(len(test), bool), actual),
        ("Policy A: reject only (keep actual rates)", funded_screen, actual),
        ("Policy B: reject + raise rates on 'ADJUST RATE' loans", funded_screen, repriced),
    ]:
        profit = realized_profit(d, rate, months, lgd)
        out[name] = {
            "loans_funded": int(funded.sum()), "share_funded": funded.mean(),
            "default_rate_funded": d[funded].mean(),
            "profit_per_$_lent": (w[funded] * profit[funded]).sum() / w[funded].sum(),
            "total_profit_$": (w[funded] * profit[funded]).sum(),
        }
    return pd.DataFrame(out).T


def main():
    train, test = get_split()
    models = load_models()
    res = recommend_df(test, models)
    print(f"Scored {len(res):,} held-out borrowers with {models[0]['name']} + {models[1]['model_name']}")
    print(f"Assumptions: LGD = {LGD:.0%}, target margin = {TARGET_MARGIN:.0%} of principal\n")

    # ---- overall result
    print("Recommended actions:")
    print(res["action"].value_counts(normalize=True).round(3).to_string())
    summary = scenarios(test, res["action"].values, res["recommended_rate"].values, LGD)
    save_table(summary.round(4), "policy_evaluation.csv")
    print("\nPolicy vs baseline (realised outcomes on the test set):")
    print(summary.round(4).to_string())

    # ---- who gets rejected / approved, actual default rates
    chk = res.groupby("action").agg(borrowers=("default", "size"), actual_default_rate=("default", "mean"),
                                   avg_pred_default=("p_default", "mean"), avg_benchmark_rate=("benchmark_rate", "mean"),
                                   avg_recommended_rate=("recommended_rate", "mean")).reindex(ACTIONS)
    save_table(chk.round(4), "actions_summary.csv")

    # ---- recommendation per borrower segment (the main table for the report)
    seg = res.groupby("segment").agg(
        borrowers=("default", "size"), actual_default_rate=("default", "mean"),
        avg_pred_default=("p_default", "mean"), avg_benchmark_rate=("benchmark_rate", "mean"))
    mix = pd.crosstab(res["segment"], res["action"], normalize="index").reindex(columns=ACTIONS, fill_value=0)
    seg = seg.join(mix.add_prefix("share_"))
    seg["avg_recommended_rate_if_funded"] = res[res["action"] != "REJECT"].groupby("segment")["recommended_rate"].mean()
    seg["avg_expected_profit_if_funded"] = res.groupby("segment")["expected_profit"].mean()
    seg["main_recommendation"] = mix.idxmax(axis=1)
    save_table(seg.round(4), "segment_recommendations.csv")
    print("\nRecommendation by segment:")
    print(seg[["borrowers", "actual_default_rate", "share_APPROVE", "share_ADJUST RATE", "share_REJECT",
               "main_recommendation"]].round(3).to_string())

    # ---- sensitivity to the assumptions
    rows = []
    for lgd in (0.4, 0.6, 0.8):
        for target in (0.01, 0.03, 0.05):
            pol = apply_policy(res["p_default"], res["benchmark_rate"], test["term"], lgd, target)
            sc = scenarios(test, pol["action"], pol["recommended_rate"], lgd)
            rows.append({"LGD": lgd, "target_margin": target,
                         "pct_approve": (pol["action"] == "APPROVE").mean(),
                         "pct_adjust": (pol["action"] == "ADJUST RATE").mean(),
                         "pct_reject": (pol["action"] == "REJECT").mean(),
                         "baseline_profit_per_$": sc.iloc[0]["profit_per_$_lent"],
                         "policyB_profit_per_$": sc.iloc[2]["profit_per_$_lent"]})
    sens = pd.DataFrame(rows)
    save_table(sens.round(4), "sensitivity_analysis.csv", index=False)
    print("\nSensitivity to assumptions:")
    print(sens.round(3).to_string(index=False))

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 4 + 0.3 * len(mix)))
    left = np.zeros(len(mix))
    for a in ACTIONS:
        ax.barh(mix.index, mix[a].values * 100, left=left, color=COLORS[a], label=a)
        left += mix[a].values * 100
    ax.set_xlabel("% of borrowers in segment"); ax.set_title("Recommended action by risk segment")
    ax.legend(loc="lower right")
    save_fig("06_actions_by_segment.png")

    plt.figure(figsize=(7, 4.5))
    for a in ACTIONS:
        sub = res.loc[res["action"] == a, "p_default"]
        if len(sub):
            plt.hist(sub, bins=40, alpha=0.7, color=COLORS[a], label=f"{a} (n={len(sub):,})")
    plt.xlabel("Predicted probability of default"); plt.ylabel("Borrowers"); plt.legend()
    plt.title("Predicted default risk by recommended action")
    save_fig("06_pd_by_action.png")

    plt.figure(figsize=(8, 4.5))
    labels = ["Baseline", "Policy A", "Policy B"]
    plt.bar(labels, summary["profit_per_$_lent"].astype(float) * 100, color=["#999999", "#3b6ea5", "#6aa84f"])
    plt.ylabel("Profit per $100 lent"); plt.title("Realised profit: baseline vs policy (test set)")
    save_fig("06_profit_comparison.png")

    keep = ["loan_amnt", "term", "annual_inc", "dti", "purpose", "int_rate", "default", "p_default",
            "benchmark_rate", "segment", "action", "recommended_rate", "required_rate", "expected_profit"]
    save_table(res[keep].round(4), "recommendations_test_set.csv", index=False)


if __name__ == "__main__":
    main()
