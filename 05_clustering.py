"""Step 5 (Module 5) - PCA, factor analysis, K-means and hierarchical clustering.

Goal: group borrowers into risk segments (using profile variables only, no outcome),
then describe each segment so the recommender can report results per segment.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from common import CLUSTER_FEATURES, MODEL_DIR, SEED, get_split, save_fig, save_table


def main():
    train, test = get_split()
    feats = CLUSTER_FEATURES
    imp = SimpleImputer(strategy="median").fit(train[feats])
    sc = StandardScaler().fit(imp.transform(train[feats]))
    Z = sc.transform(imp.transform(train[feats]))
    n, p = Z.shape

    # ---- is the data suitable for factor analysis / PCA? (Bartlett sphericity test)
    R = np.corrcoef(Z, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    p_val = stats.chi2.sf(chi2, p * (p - 1) / 2)
    print(f"Bartlett's test of sphericity: chi2 = {chi2:,.0f}, p = {p_val:.3g}")

    # ---- PCA
    pca_full = PCA().fit(Z)
    eig, cum = pca_full.explained_variance_, np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= 0.80) + 1)
    print(f"Components needed for 80% variance: {n_comp} (Kaiser rule: {int((eig > 1).sum())} eigenvalues > 1)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, p + 1), eig, marker="o"); ax[0].axhline(1, color="red", ls="--")
    ax[0].set_title("Scree plot (red = Kaiser line)"); ax[0].set_xlabel("Component"); ax[0].set_ylabel("Eigenvalue")
    ax[1].plot(range(1, p + 1), cum * 100, marker="o"); ax[1].axhline(80, color="red", ls="--")
    ax[1].set_title("Cumulative variance explained (%)"); ax[1].set_xlabel("Component")
    save_fig("05_pca_scree.png")

    pca = PCA(n_components=n_comp).fit(Z)
    cols = [f"PC{i + 1}" for i in range(n_comp)]
    save_table(pd.DataFrame(pca.components_.T, index=feats, columns=cols).round(3), "pca_loadings.csv")
    P = pca.transform(Z)

    # ---- factor analysis (varimax rotation)
    n_fac = max(2, min(int((eig > 1).sum()), 5))
    fa = FactorAnalysis(n_components=n_fac, rotation="varimax", random_state=SEED).fit(Z)
    save_table(pd.DataFrame(fa.components_.T, index=feats, columns=[f"Factor{i + 1}" for i in range(n_fac)]).round(3),
               "factor_loadings.csv")

    # ---- K-means: elbow + silhouette
    sil_idx = np.random.default_rng(SEED).choice(len(P), size=min(10000, len(P)), replace=False)
    ks, inertia, sil = range(2, 9), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(P)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(P[sil_idx], km.labels_[sil_idx]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(ks), inertia, marker="o"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
    ax[1].plot(list(ks), sil, marker="o"); ax[1].set_title("Average silhouette"); ax[1].set_xlabel("k")
    save_fig("05_kmeans_elbow_silhouette.png")
    k_range = [k for k in ks if 3 <= k <= 6]          # at least 3 segments, at most 6 to stay readable
    best_k = max(k_range, key=lambda k: sil[list(ks).index(k)])
    print(f"Chosen k = {best_k} (silhouette {sil[list(ks).index(best_k)]:.3f})")
    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED).fit(P)

    # ---- hierarchical clustering (Ward) on a 2,000-row subsample
    h_idx = np.random.default_rng(SEED).choice(len(P), size=2000, replace=False)
    Lk = linkage(P[h_idx], method="ward")
    plt.figure(figsize=(10, 4.5))
    dendrogram(Lk, truncate_mode="lastp", p=30, no_labels=True)
    plt.title("Hierarchical clustering (Ward), 2,000-borrower sample"); plt.ylabel("Distance")
    save_fig("05_dendrogram.png")
    ari = adjusted_rand_score(fcluster(Lk, best_k, criterion="maxclust"), km.predict(P[h_idx]))
    print(f"Agreement between hierarchical and K-means (adjusted Rand index): {ari:.3f}")

    # ---- profile the clusters using the training data
    train = train.copy()
    train["cluster"] = km.labels_
    order = train.groupby("cluster")["default"].mean().sort_values().index.tolist()
    names = {c: f"S{i + 1}" for i, c in enumerate(order)}      # S1 = lowest default rate
    centers = pd.DataFrame(sc.transform(imp.transform(train[feats])), columns=feats)
    centers["cluster"] = train["cluster"].values
    z = centers.groupby("cluster").mean()

    def traits(c):
        top = z.loc[c].reindex(z.loc[c].abs().sort_values(ascending=False).index)[:3]
        return ", ".join(f"{'high' if v > 0 else 'low'} {k}" for k, v in top.items())

    prof = train.groupby("cluster").agg(
        size=("default", "size"), default_rate=("default", "mean"), avg_loan=("loan_amnt", "mean"),
        avg_income=("annual_inc", "mean"), avg_dti=("dti", "mean"), avg_revol_util=("revol_util", "mean"),
        avg_int_rate=("int_rate", "mean"), pct_60m=("term", lambda s: (s == 60).mean()))
    prof["share"] = prof["size"] / len(train)
    prof["defining_traits"] = [traits(c) for c in prof.index]
    prof.index = [names[c] for c in prof.index]
    prof = prof.sort_index()
    save_table(prof.round(3), "cluster_profile.csv")
    print("\n" + prof[["size", "default_rate", "avg_dti", "avg_income", "defining_traits"]].round(3).to_string())

    # heatmap of standardised cluster centres
    zz = z.copy(); zz.index = [names[c] for c in zz.index]; zz = zz.sort_index()
    fig, ax = plt.subplots(figsize=(10, 4 + 0.3 * len(zz)))
    im = ax.imshow(zz.values, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=60, ha="right")
    ax.set_yticks(range(len(zz))); ax.set_yticklabels(zz.index)
    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            ax.text(j, i, f"{zz.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im); ax.set_title("Segment profiles (standardised feature means)")
    save_fig("05_segment_heatmap.png")

    # 2-D view of the segments
    s = np.random.default_rng(SEED).choice(len(P), size=min(6000, len(P)), replace=False)
    plt.figure(figsize=(6.5, 5.5))
    for c in order:
        m = km.labels_[s] == c
        plt.scatter(P[s][m, 0], P[s][m, 1], s=5, alpha=0.4, label=names[c])
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(markerscale=3); plt.title("Segments on the first two principal components")
    save_fig("05_segments_pca_scatter.png")

    joblib.dump({"features": feats, "imputer": imp, "scaler": sc, "pca": pca, "kmeans": km,
                 "segment_names": names}, MODEL_DIR / "cluster_model.joblib")
    print("  saved model : outputs/models/cluster_model.joblib")


if __name__ == "__main__":
    main()
