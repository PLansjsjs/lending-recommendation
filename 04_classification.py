"""Step 4 (Module 4) - predict default with five classifiers and compare them.

Models : Logistic Regression, Decision Tree, KNN, Naive Bayes, LDA
Metrics: confusion matrix, accuracy, precision, recall, F1, ROC-AUC
The best model (by cross-validated AUC) is saved for the recommender.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from common import (MODEL_DIR, SEED, TAB_DIR, get_feature_lists, get_split, make_preprocessor,
                    save_fig, save_table)

# Naive Bayes probabilities are not well calibrated, so it is compared but never
# used to price loans. The class_weight model is for the imbalance discussion only.
NOT_FOR_PRICING = {"Naive Bayes", "Logistic Regression (balanced)"}


def pipe(model, num, cat, scale=True):
    return Pipeline([("pre", make_preprocessor(num, cat, scale=scale)), ("model", model)])


def metrics_rows(name, y, proba, extra):
    fpr, tpr, thr = roc_curve(y, proba)
    youden = float(thr[np.argmax(tpr - fpr)])
    auc = roc_auc_score(y, proba)
    rows = []
    for label, t in [("0.50", 0.5), ("Youden (best ROC point)", youden)]:
        tn, fp, fn, tp = confusion_matrix(y, (proba >= t).astype(int)).ravel()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        rows.append({"model": name, "threshold_rule": label, "threshold": round(t, 4), "AUC": auc,
                     "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
                     "specificity": tn / (tn + fp), "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                     "TN": tn, "FP": fp, "FN": fn, "TP": tp, **extra})
    return rows, youden


def main():
    train, test = get_split()
    num, cat = get_feature_lists()
    Xtr, ytr = train[num + cat], train["default"]
    Xte, yte = test[num + cat], test["default"]
    print(f"Train {len(Xtr):,} rows | Test {len(Xte):,} rows | default rate {ytr.mean():.1%}")

    # ---- tune the tree and KNN with 3-fold cross-validation
    tree_gs = GridSearchCV(pipe(DecisionTreeClassifier(random_state=SEED), num, cat, scale=False),
                           {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [100, 300]},
                           cv=3, scoring="roc_auc").fit(Xtr, ytr)
    sub = np.random.default_rng(SEED).choice(len(Xtr), size=min(12000, len(Xtr)), replace=False)
    knn_gs = GridSearchCV(pipe(KNeighborsClassifier(), num, cat),
                          {"model__n_neighbors": [15, 31, 61, 101]},
                          cv=3, scoring="roc_auc").fit(Xtr.iloc[sub], ytr.iloc[sub])
    print("Best tree params:", tree_gs.best_params_)
    print("Best KNN params :", knn_gs.best_params_)

    models = {
        "Logistic Regression": (pipe(LogisticRegression(max_iter=2000), num, cat), ""),
        "Decision Tree": (pipe(DecisionTreeClassifier(
            max_depth=tree_gs.best_params_["model__max_depth"],
            min_samples_leaf=tree_gs.best_params_["model__min_samples_leaf"], random_state=SEED),
            num, cat, scale=False), str(tree_gs.best_params_)),
        "KNN": (pipe(KNeighborsClassifier(n_neighbors=knn_gs.best_params_["model__n_neighbors"]), num, cat),
                str(knn_gs.best_params_)),
        "Naive Bayes": (pipe(GaussianNB(), num, cat), ""),
        "LDA": (pipe(LinearDiscriminantAnalysis(), num, cat), ""),
        "Logistic Regression (balanced)": (
            pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), num, cat), "class_weight=balanced"),
    }

    all_rows, probas, thresholds, cv_auc = [], {}, {}, {}
    for name, (model, params) in models.items():
        print(f"Fitting {name} ...")
        cv = cross_val_score(model, Xtr, ytr, cv=3, scoring="roc_auc")
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        rows, youden = metrics_rows(name, yte, proba,
                                    {"cv_AUC_mean": cv.mean(), "cv_AUC_std": cv.std(), "params": params})
        all_rows += rows
        probas[name], thresholds[name], cv_auc[name] = proba, youden, cv.mean()

    res = pd.DataFrame(all_rows)
    save_table(res.round(4), "classifier_comparison.csv", index=False)
    show = res[["model", "threshold_rule", "AUC", "accuracy", "precision", "recall", "F1", "cv_AUC_mean"]]
    print("\n" + show.round(3).to_string(index=False))
    print("\nWith ~20% defaults, accuracy at 0.50 is misleading: look at recall, AUC and the Youden rows.")

    # ---- ROC curves
    plt.figure(figsize=(6.5, 5.5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
    plt.legend(fontsize=8)
    save_fig("04_roc_curves.png")

    # ---- confusion matrices at the Youden threshold
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), probas):
        cm = confusion_matrix(yte, (probas[name] >= thresholds[name]).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred paid", "Pred default"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual paid", "Actual default"])
        ax.set_title(f"{name}\n(threshold {thresholds[name]:.3f})", fontsize=9)
    save_fig("04_confusion_matrices.png")

    # ---- best model for pricing (highest CV AUC among eligible models)
    eligible = {k: v for k, v in cv_auc.items() if k not in NOT_FOR_PRICING}
    best_name = max(eligible, key=eligible.get)
    best_pipe = models[best_name][0]
    print(f"\nBest model for the recommender: {best_name} (CV AUC {eligible[best_name]:.4f})")
    joblib.dump({"name": best_name, "pipeline": best_pipe, "features": num + cat,
                 "youden_threshold": thresholds[best_name]}, MODEL_DIR / "best_classifier.joblib")

    # calibration: are predicted probabilities close to observed default rates?
    p = probas[best_name]
    bins = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"p": p, "y": yte.values}).groupby(bins, observed=True).agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    save_table(cal.round(4), "calibration_best_model.csv")
    plt.figure(figsize=(5, 5))
    plt.plot(cal["mean_predicted"], cal["observed_rate"], marker="o")
    plt.plot([0, cal["mean_predicted"].max()], [0, cal["mean_predicted"].max()], "k--")
    plt.xlabel("Mean predicted default probability"); plt.ylabel("Observed default rate")
    plt.title(f"Calibration by decile: {best_name}")
    save_fig("04_calibration.png")

    # ---- interpretation: logistic regression with p-values (statsmodels)
    pre_sm = make_preprocessor(num, cat, scale=True, drop_first=True)
    Xs = pd.DataFrame(pre_sm.fit_transform(Xtr),
                      columns=[n.split("__", 1)[1] for n in pre_sm.get_feature_names_out()])
    logit = sm.Logit(ytr.values, sm.add_constant(Xs)).fit(disp=0, maxiter=200)
    (TAB_DIR / "logit_summary.txt").write_text(logit.summary().as_text(), encoding="utf-8")
    ci = logit.conf_int()
    odds = pd.DataFrame({"coef": logit.params, "odds_ratio": np.exp(logit.params), "p_value": logit.pvalues,
                         "OR_ci_low": np.exp(ci[0]), "OR_ci_high": np.exp(ci[1])})
    save_table(odds.round(4).sort_values("p_value"), "logit_odds_ratios.csv")
    print(f"Logit pseudo R-squared (McFadden): {logit.prsquared:.4f}")
    print("Numeric variables are standardised: odds ratio = change in odds per +1 standard deviation.")

    # ---- interpretation: decision-tree rules
    tree_pipe = models["Decision Tree"][0]
    names = [n.split("__", 1)[1] for n in tree_pipe.named_steps["pre"].get_feature_names_out()]
    tree = tree_pipe.named_steps["model"]
    (TAB_DIR / "tree_rules.txt").write_text(export_text(tree, feature_names=names, max_depth=3), encoding="utf-8")
    plt.figure(figsize=(22, 9))
    plot_tree(tree, feature_names=names, class_names=["paid", "default"], max_depth=3,
              filled=True, fontsize=7, proportion=True)
    save_fig("04_decision_tree.png")


if __name__ == "__main__":
    main()
