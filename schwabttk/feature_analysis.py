# schwabttk/feature_analysis.py
"""
Feature Analysis & Selection
==============================
Consumes the output of price_features.compute_all_features() and provides
tools to understand, rank, and select the most informative and independent
features for downstream use in trend_score.py.

This module is stateful — PCA and mutual information require fitting on
historical data. Results should be inspected before being passed downstream.

Workflow
--------
1.  build_feature_matrix()    — extract numeric trend-relevant features
2.  standardize_features()    — rolling z-score (no lookahead)
3.  correlation_analysis()    — pairwise correlations + clustering
4.  variance_analysis()       — which features carry the most variance
5.  run_pca()                 — discover independent dimensions of trend
6.  pairwise_mi_matrix()      — MI between features (independence, not prediction)
7.  select_features()         — variance floor + iterative MI-cluster / κ elimination
8.  feature_report()          — consolidated summary of all analyses
9.  run_full_analysis()       — convenience: steps 1-8 in one call

Dependency
----------
    price_features.compute_all_features(df) → df_features
    feature_analysis.run_full_analysis(df_features) → FeatureAnalysisResult
    trend_score.compute_trend_score(df_features, result.selected_features, ...) → df_scored
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import percentileofscore
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class FeatureAnalysisResult:
    feature_cols      : list = field(default_factory=list)
    X                 : Optional[pd.DataFrame] = None
    X_std             : Optional[pd.DataFrame] = None
    corr_matrix       : Optional[pd.DataFrame] = None
    corr_clusters     : Optional[dict] = None
    variance_rank     : Optional[dict] = None
    pca               : Optional[dict] = None
    pairwise_mi       : Optional[pd.DataFrame] = None   # ← replaces mi_scores
    mi_scores         : None = None                     # ← kept as None, deprecated
    selected_features : list = field(default_factory=list)
    selection_detail  : Optional[dict] = None           # full select_features() output
    selected_corr     : Optional[pd.DataFrame] = None   # corr submatrix of selected
    selected_pmi      : Optional[pd.DataFrame] = None   # pairwise MI submatrix of selected
    periods           : list = field(default_factory=lambda: [21, 64, 128])
    std_window        : int = 252
    target_col        : str = "pairwise_MI"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def _candidate_feature_cols(df: pd.DataFrame, periods: list) -> list:
    """
    Return list of numeric trend-relevant feature columns present in df.

    Inclusion criteria
    ------------------
    - Continuous numeric
    - Signed (carries directional information)
    - Not raw price or volume levels
    - Not categorical (Pattern_*, Regime, is_* flags, doji_threshold)

    Per-bar (period-independent)
    ----------------------------
    CLV, BodyRatio, UpperRatio, LowerRatio, Conviction

    Per-period rolling features
    ---------------------------
    TM, ER, DC, AC1, AC5, AC11, CorrRR,
    AvgRunPos, AvgRunNeg, DomRun,
    RVOL, VTC, VWR, VMD, PVE, PVE_pct, EMA_slope, MOM
    """
    per_bar = [
        # "r_day", used as MI target, not as feature
        "CLV",
        "BodyRatio",
        "UpperRatio",
        "LowerRatio",
        "Conviction",
    ]
    per_period = []
    for p in periods:
        sfx = f"_{p}"
        per_period += [
            # f"R{sfx}", information already in TM, ER, MOM, DC
            f"TM{sfx}",
            f"ER{sfx}",
            f"DC{sfx}",
            f"AC1{sfx}",
            f"AC5{sfx}",
            f"AC11{sfx}",
            f"CorrRR{sfx}",
            f"AvgRunPos{sfx}",
            f"AvgRunNeg{sfx}",
            f"DomRun{sfx}",
            f"RVOL{sfx}",
            f"VTC{sfx}",
            f"VWR{sfx}",
            f"VMD{sfx}",
            f"PVE{sfx}",
            f"PVE_pct{sfx}",
            f"EMA_slope{sfx}",
            f"MOM{sfx}",
        ]

    candidates = per_bar + per_period
    present    = [c for c in candidates if c in df.columns]
    missing    = [c for c in candidates if c not in df.columns]
    if missing:
        print(f"[WARN] {len(missing)} candidate columns not found — "
              f"run compute_all_features() first:\n  {missing}")
    return present


def build_feature_matrix(df: pd.DataFrame,
                         periods: list = None,
                         min_obs: int = 128) -> pd.DataFrame:
    """
    Extract and clean the numeric feature matrix.

    Parameters
    ----------
    df      : output of compute_all_features()
    periods : must match what was used in compute_all_features()
    min_obs : minimum valid observations required per column (default 128)

    Returns
    -------
    pd.DataFrame — numeric features only, NaN rows dropped
    """
    if periods is None:
        periods = [21, 64, 128]

    cols = _candidate_feature_cols(df, periods)
    X    = df[cols].copy()

    # Drop columns with insufficient valid observations
    valid = [c for c in X.columns if X[c].notna().sum() >= min_obs]
    dropped = set(cols) - set(valid)
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} columns with "
              f"< {min_obs} valid observations: {sorted(dropped)}")

    X = X[valid].dropna()
    print(f"[INFO] Feature matrix: {X.shape[0]} rows × {X.shape[1]} cols")
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score standardization — no lookahead bias.
    Uses only past observations within each rolling window.
    Returns NaN for the first `window` observations.
    """
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std().replace(0, np.nan)
    return (series - mu) / sig


def standardize_features(X: pd.DataFrame,
                          window: int = 252) -> pd.DataFrame:
    """
    Apply rolling z-score to each feature column independently.

    Parameters
    ----------
    X      : raw feature matrix from build_feature_matrix()
    window : rolling window (default 252 ≈ 1 trading year)

    Returns
    -------
    pd.DataFrame — same shape as X, standardized values
    """
    X_std = X.apply(lambda col: rolling_zscore(col, window))
    n_nan = X_std.isna().any(axis=1).sum()
    print(f"[INFO] Standardization complete | "
          f"window={window} | NaN rows: {n_nan}")
    return X_std


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def correlation_analysis(X: pd.DataFrame,
                          method: str = "pearson",
                          cluster_threshold: float = 0.70) -> dict:
    """
    Compute pairwise feature correlations and cluster correlated features.

    Clustering method: hierarchical agglomerative clustering on
    the distance matrix (1 - |corr|). Features within the same cluster
    are interchangeable — only one per cluster is needed for modeling.

    Parameters
    ----------
    X                 : feature matrix (raw or standardized)
    method            : 'pearson' or 'spearman' (default 'pearson')
    cluster_threshold : |corr| above this → same cluster (default 0.70)

    Returns
    -------
    dict:
        corr_matrix    : pd.DataFrame — full pairwise correlation matrix
        abs_corr       : pd.DataFrame — absolute correlations
        high_corr_pairs: list of (feat_a, feat_b, corr) for |corr| > threshold
        clusters       : dict {cluster_id: [feature names]}
        cluster_labels : pd.Series — cluster id per feature
        n_clusters     : int
    """
    X_clean = X.dropna()
    corr    = X_clean.corr(method=method)
    abs_c   = corr.abs()

    # High correlation pairs
    mask  = np.triu(np.ones(abs_c.shape, dtype=bool), k=1)
    upper = abs_c.where(mask)
    high_pairs = [
        (col, row, round(float(corr.loc[row, col]), 4))
        for col in upper.columns
        for row in upper.index
        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > cluster_threshold
    ]
    high_pairs.sort(key=lambda x: -abs(x[2]))

    # Hierarchical clustering on distance matrix
    dist_matrix = 1 - abs_c.fillna(0).clip(0, 1)
    condensed   = squareform(dist_matrix.values, checks=False)
    linkage     = hierarchy.linkage(condensed, method="average")
    labels      = hierarchy.fcluster(
        linkage, t=1 - cluster_threshold, criterion="distance"
    )

    cluster_labels = pd.Series(labels, index=corr.columns, name="cluster")
    clusters = {}
    for feat, cid in cluster_labels.items():
        clusters.setdefault(int(cid), []).append(feat)

    n_clusters = len(clusters)
    print(f"[INFO] Correlation analysis | method={method} | "
          f"threshold={cluster_threshold}")
    print(f"[INFO] {len(high_pairs)} high-correlation pairs | "
          f"{n_clusters} clusters")

    return {
        "corr_matrix"    : corr,
        "abs_corr"       : abs_c,
        "high_corr_pairs": high_pairs,
        "clusters"       : clusters,
        "cluster_labels" : cluster_labels,
        "n_clusters"     : n_clusters,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VARIANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def variance_analysis(X: pd.DataFrame, rolling_window: int = 252) -> dict:
    """
    Rank features by the amount of variance they carry.

    Two complementary measures:
    - Global variance  : var(feature) across all observations
    - Rolling variance : mean of rolling(window).var() — how much
                         the feature changes over time (time-varying signal)

    Note: on rolling z-scores every feature has variance ≈ 1, so the
    ranking mainly detects features that are near-constant or whose
    rolling std collapses (degenerate z-scores).

    Parameters
    ----------
    X              : feature matrix
    rolling_window : window for time-varying variance estimate (252)

    Returns
    -------
    dict:
        global_var   : pd.Series — global variance per feature (ranked)
        rolling_var  : pd.Series — mean rolling variance per feature (ranked)
        combined_rank: pd.Series — mean rank across both measures (ranked)
    """
    X_clean = X.dropna()

    global_var  = X_clean.var().sort_values(ascending=False)
    rolling_var = (
        X_clean
        .rolling(rolling_window)
        .var()
        .mean()
        .sort_values(ascending=False)
    )

    # Rank both (1 = highest variance) and average
    gv_rank = global_var.rank(ascending=False)
    rv_rank = rolling_var.rank(ascending=False)
    combined = ((gv_rank + rv_rank) / 2).sort_values()

    print(f"[INFO] Variance analysis | rolling_window={rolling_window}")
    print(f"[INFO] Top-10 by combined rank:")
    for feat in combined.head(10).index:
        print(f"  {feat:<28} global_var={global_var.get(feat, np.nan):.6f}  "
              f"rolling_var={rolling_var.get(feat, np.nan):.6f}")

    return {
        "global_var"   : global_var,
        "rolling_var"  : rolling_var,
        "combined_rank": combined,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PCA
# ═══════════════════════════════════════════════════════════════════════════════
def run_pca(X_std: pd.DataFrame,
            n_components: int = None,
            variance_threshold: float = 0.90) -> dict:
    """
    Fit PCA on the standardized feature matrix.

    Discovers the independent dimensions (principal components) that
    explain most of the variance across all features. Each PC represents
    a distinct aspect of trend that is orthogonal to all other PCs.

    Parameters
    ----------
    X_std              : standardized feature matrix (NaNs dropped)
    n_components       : fixed number of PCs (None = auto)
    variance_threshold : cumulative variance threshold for auto selection

    Returns
    -------
    dict:
        pca               : fitted sklearn PCA object
        loadings          : pd.DataFrame (features × PCs)
        explained         : pd.Series — fraction of variance per PC
        cumulative        : pd.Series — cumulative variance fraction
        n_components      : int — number of PCs selected
        scores            : pd.DataFrame — PC scores (observations × PCs)
        feature_importance: dict {PC: top-5 features by |loading|}
    """
    X_clean = X_std.dropna()

    if n_components is None:
        pca_full = PCA().fit(X_clean)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumvar >= variance_threshold) + 1)
        print(f"[INFO] Auto-selected {n_components} PCs → "
              f"{cumvar[n_components-1]*100:.1f}% variance explained")

    pca       = PCA(n_components=n_components)
    scores    = pca.fit_transform(X_clean)
    pc_labels = [f"PC{i+1}" for i in range(n_components)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X_clean.columns,
        columns=pc_labels
    )
    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=pc_labels,
        name="variance_explained"
    )

    feature_importance = {}
    for pc in pc_labels:
        top5 = loadings[pc].abs().sort_values(ascending=False).head(5)
        feature_importance[pc] = {
            feat: {
                "loading"    : round(float(loadings.loc[feat, pc]), 4),
                "abs_loading": round(float(top5[feat]), 4),
            }
            for feat in top5.index
        }

    scores_df = pd.DataFrame(
        scores, index=X_clean.index, columns=pc_labels
    )

    print(f"[INFO] PCA complete | {n_components} components")
    for pc in pc_labels:
        var = explained[pc]
        cum = explained.cumsum()[pc]
        top = list(feature_importance[pc].keys())[:3]
        print(f"  {pc}: {var*100:.1f}% var  "
              f"(cum {cum*100:.1f}%)  top: {top}")

    return {
        "pca"               : pca,
        "loadings"          : loadings,
        "explained"         : explained,
        "cumulative"        : explained.cumsum(),
        "n_components"      : n_components,
        "scores"            : scores_df,
        "feature_importance": feature_importance,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PAIRWISE MI (feature independence, not prediction)
# NOTE: mutual_information_vs_target() is NOT part of the main pipeline.
# Feature selection here is a DESCRIPTION task, not a PREDICTION task.
# It is kept as an optional standalone tool for separate alpha research.
# ═══════════════════════════════════════════════════════════════════════════════
def mutual_information_vs_target(X_std: pd.DataFrame,
                                  target_col: str = None,
                                  target_series: pd.Series = None,
                                  forward_shift: int = 1,
                                  n_neighbors: int = 5) -> pd.Series:
    """
    [OPTIONAL / STANDALONE] Compute MI between each feature and a target.

    NOT used in run_full_analysis() — kept for separate alpha research.

    For trend description/scoring, use pairwise_mi_matrix() instead.
    Using a target here introduces bias: R_p columns share most of their
    bars with a forward-shifted R_p target, inflating their MI scores.
    r_day is the least-biased target if you do use this function.

    Parameters
    ----------
    X_std         : standardized feature matrix (NaNs dropped)
    target_col    : column in X_std to use as target (bias risk with R_p)
    target_series : external pd.Series — preferred over target_col
                    (pass df["r_day"] to avoid overlap bias)
    forward_shift : bars ahead for the target (default 1)
    n_neighbors   : k for k-NN MI estimator (default 5)

    Returns
    -------
    pd.Series — MI score per feature vs target (sorted descending)
    """
    X_clean = X_std.dropna()

    if target_series is not None:
        target = target_series.shift(-forward_shift).dropna()
        target.name = target.name or "target"
    elif target_col is not None:
        if target_col not in X_clean.columns:
            raise ValueError(
                f"target_col '{target_col}' not in X_std. "
                f"Pass it via target_series instead."
            )
        target = X_clean[target_col].shift(-forward_shift).dropna()
    else:
        raise ValueError("Provide either target_col or target_series.")

    X_mi   = X_clean.loc[X_clean.index.intersection(target.index)].fillna(0)
    target = target.loc[X_mi.index]

    mi_raw    = mutual_info_regression(
        X_mi, target, n_neighbors=n_neighbors, random_state=42
    )
    mi_scores = pd.Series(mi_raw, index=X_mi.columns).sort_values(
        ascending=False
    )

    tname = target.name if target.name else "external"
    print(f"[INFO] MI vs target={tname} | shift={forward_shift} | top-10:")
    for feat, score in mi_scores.head(10).items():
        print(f"  {feat:<28} MI={score:.4f}")

    return mi_scores


def _pairwise_mi(X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Symmetrized k-NN pairwise MI matrix over all columns of X."""
    cols   = list(X.columns)
    n      = len(cols)
    mi_mat = np.zeros((n, n))
    for i, feat in enumerate(cols):
        mi_mat[i, :] = mutual_info_regression(
            X, X[feat], n_neighbors=n_neighbors, random_state=42
        )
    mi_mat = (mi_mat + mi_mat.T) / 2
    return pd.DataFrame(mi_mat, index=cols, columns=cols)


def pairwise_mi_matrix(X_std: pd.DataFrame,
                        n_neighbors: int = 5,
                        max_features: int = 64) -> pd.DataFrame:
    """
    Compute pairwise mutual information matrix between features.

    MI(feature_i, feature_j) measures how much information
    feature_i and feature_j share — regardless of any target.
    Low MI → complementary. High MI → redundant.

    Parameters
    ----------
    X_std        : standardized feature matrix (NaNs dropped)
    n_neighbors  : k for k-NN MI estimator (default 5)
    max_features : cap to avoid O(n²) explosion on large sets (default 64).
                   Set ≥ number of features so that select_features()
                   sees every feature in its first iteration.

    Returns
    -------
    pd.DataFrame — symmetric MI matrix (features × features)
    """
    X_clean = X_std.dropna()

    if len(X_clean.columns) > max_features:
        print(f"[WARN] Capping pairwise MI to top {max_features} features "
              f"by variance (from {len(X_clean.columns)} total) — "
              f"uncovered features cannot be clustered in iteration 1")
        var_rank = X_clean.var().sort_values(ascending=False)
        X_clean  = X_clean[var_rank.head(max_features).index]

    n = X_clean.shape[1]
    print(f"[INFO] Computing pairwise MI matrix "
          f"({n} × {n} = {n*n} pairs)...")
    mi_df = _pairwise_mi(X_clean, n_neighbors=n_neighbors)
    print(f"[INFO] Pairwise MI matrix complete.")
    return mi_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def condition_number(X: pd.DataFrame) -> float:
    """
    Condition number of the column-scaled feature matrix:
        κ = sqrt(λ_max / λ_min) of the correlation matrix.

    Equivalent to the Belsley condition index of unit-scaled X, so the
    usual rule of thumb applies (κ > 30 → serious collinearity).
    Returns inf if the matrix is singular, 1.0 for a single column.
    """
    X = X.dropna()
    if X.shape[1] <= 1:
        return 1.0
    corr = np.corrcoef(X.values, rowvar=False)
    if not np.all(np.isfinite(corr)):
        return np.inf
    eig = np.linalg.eigvalsh(corr)
    lam_min, lam_max = float(eig.min()), float(eig.max())
    if lam_min <= 1e-12:
        return np.inf
    return float(np.sqrt(lam_max / lam_min))


def _kappa_from_corr(C: np.ndarray) -> float:
    """κ = sqrt(λ_max / λ_min) of a correlation (sub)matrix."""
    if C.shape[0] <= 1:
        return 1.0
    if not np.all(np.isfinite(C)):
        return np.inf
    eig = np.linalg.eigvalsh(C)
    lam_min, lam_max = float(eig.min()), float(eig.max())
    if lam_min <= 1e-12:
        return np.inf
    return float(np.sqrt(lam_max / lam_min))


def communality(pca_result: dict, X: pd.DataFrame) -> pd.Series:
    """
    Fraction of each feature's variance explained by the retained PCs:
        h²_f = Σ_k λ_k · v_fk² / var(f)

    Unlike max|loading|, invariant to PC ordering, sign flips and rotation
    within the retained subspace — so comparable across rolling windows.
    """
    pca = pca_result["pca"]
    L   = pca_result["loadings"]
    lam = pca.explained_variance_
    var = X[L.index].dropna().var()
    h2  = (L.values ** 2 * lam).sum(axis=1) / var.values
    return pd.Series(np.clip(h2, 0, 1), index=L.index, name="communality")


def vif_scores(X: pd.DataFrame) -> pd.Series:
    """
    Variance inflation factors: VIF_j = [R⁻¹]_jj, R = correlation matrix.
    Uses the pseudo-inverse so a singular set returns large VIFs, not an error.
    """
    X = X.dropna()
    if X.shape[1] <= 1:
        return pd.Series(1.0, index=X.columns, name="VIF")
    corr = np.corrcoef(X.values, rowvar=False)
    inv  = np.linalg.pinv(corr)
    return pd.Series(np.diag(inv), index=X.columns, name="VIF") \
             .sort_values(ascending=False)


def select_features(X_std: pd.DataFrame,
                    pca_result: dict,
                    variance_rank: dict,
                    pairwise_mi: pd.DataFrame,
                    pmi_quantile: float = 0.50,
                    var_floor_fraction: float = 0.10,
                    max_condition_number: float = 30.0,
                    min_pc_loading: float = 0.10,
                    pairwise_mi_neighbors: int = 5,
                    pc_cost_method: str = "max_loading") -> dict:
    """
    Top-down feature selection via unified MI+κ iterative elimination.

    Note on pairwise MI: MI(f_i, f_j) depends only on the pair, so the
    matrix on the surviving set is exactly a submatrix of the initial one
    (verified to 1e-15). The loop therefore subsets rather than recomputes;
    only the data-driven threshold (quantile over surviving pairs) changes
    between iterations. Features missing from `pairwise_mi` (e.g. because
    of max_features capping) trigger one recomputation up front.

    pc_cost_method : "max_loading" — max |loading| over retained PCs
                     "communality" — Σ_k λ_k v_fk² / var(f); rotation- and
                     order-invariant, preferred for rolling comparisons.

    Algorithm
    ---------
    Step 1 — Variance floor:
        Drop features below var_floor_fraction × median(variance).

    Step 2 — κ baseline:
        Condition number of the surviving feature matrix.
        If already ≤ max_condition_number → done.

    Step 3 — Unified elimination loop (repeats until κ ≤ target):
        a. Pairwise MI restricted to CURRENT surviving features
        b. Recompute κ; stop if target met
        c. Re-identify redundancy clusters from fresh pairwise MI
           (edge if MI > pmi_quantile of positive off-diagonal MI)
        d. Score every feature in a cluster (singletons never dropped):
             drop_score(f) = κ_improvement(f) / pc_loading_cost(f)
        e. Drop the single highest-scored feature
        f. Repeat

    Parameters
    ----------
    X_std                : standardized feature matrix (NaNs dropped)
    pca_result           : output of run_pca()
    variance_rank        : output of variance_analysis()
    pairwise_mi          : initial pairwise MI matrix (first iteration only)
    pmi_quantile         : MI quantile for cluster formation (0.50)
    var_floor_fraction   : variance floor as fraction of median (0.10)
    max_condition_number : target κ ceiling (30.0)
    min_pc_loading       : floor for pc_loading_cost (0.10)
    pairwise_mi_neighbors: k for k-NN MI estimator on recomputation (5)

    Returns
    -------
    dict:
        selected_features, dropped_features, kappa_baseline, kappa_final,
        kappa_path, vif_final, selected_pmi, var_floor, n_iterations
    """
    X_clean   = X_std.dropna()
    loadings  = pca_result["loadings"]
    gvar      = variance_rank["global_var"]

    if pc_cost_method == "communality":
        _cost_vals = communality(pca_result, X_clean)
    elif pc_cost_method == "max_loading":
        _cost_vals = loadings.abs().max(axis=1)
    else:
        raise ValueError(f"unknown pc_cost_method '{pc_cost_method}'")

    def _pc_cost(f):
        c = float(_cost_vals[f]) if f in _cost_vals.index else min_pc_loading
        return max(c, min_pc_loading)

    # Correlation matrix computed once; κ of any subset = κ of its submatrix
    _cols  = list(X_clean.columns)
    _pos   = {c: i for i, c in enumerate(_cols)}
    _Cfull = np.corrcoef(X_clean.values, rowvar=False)

    def _kappa(feats):
        idx = [_pos[f] for f in feats]
        return _kappa_from_corr(_Cfull[np.ix_(idx, idx)])

    # ── Step 1: Variance floor ────────────────────────────────────────────
    median_var   = float(gvar.median())
    var_floor    = var_floor_fraction * median_var
    surviving    = [f for f in X_clean.columns
                    if float(gvar.get(f, 0)) > var_floor]
    flat_dropped = [f for f in X_clean.columns if f not in surviving]

    print(f"[INFO] Variance floor: {var_floor:.6f} "
          f"({var_floor_fraction:.0%} × median={median_var:.6f})")
    print(f"[INFO] Dropped {len(flat_dropped)} flat features")
    if flat_dropped:
        print(f"  {flat_dropped}")
    print(f"[INFO] Surviving after variance floor: {len(surviving)}")

    # ── Step 2: κ baseline ────────────────────────────────────────────────
    kappa_baseline = _kappa(surviving)
    print(f"\n[INFO] κ baseline: {kappa_baseline:.2f} "
          f"(target ≤ {max_condition_number})")

    kappa_path       = []
    dropped_features = list(flat_dropped)
    n_iterations     = 0
    current_pmi      = pairwise_mi

    # Fill the gap once if the initial PMI does not cover every survivor
    uncovered = [f for f in surviving if f not in pairwise_mi.index]
    if uncovered and kappa_baseline > max_condition_number:
        print(f"[INFO] {len(uncovered)} survivors missing from pairwise MI — "
              f"recomputing once on {len(surviving)} features")
        pairwise_mi = current_pmi = _pairwise_mi(
            X_clean[surviving], n_neighbors=pairwise_mi_neighbors
        )

    if kappa_baseline <= max_condition_number:
        print(f"[INFO] Already within target — no elimination needed")
    else:
        # ── Step 3: Unified elimination loop ─────────────────────────────
        div = "─" * 60

        while True:
            n_iterations += 1
            kappa_now     = _kappa(surviving)

            print(f"\n{div}")
            print(f"[ITER {n_iterations}] κ={kappa_now:.2f} | "
                  f"features={len(surviving)}")

            if kappa_now <= max_condition_number:
                print(f"  κ ≤ {max_condition_number} ✓ — stopping")
                break

            if len(surviving) <= 1:
                print(f"  [WARN] Only 1 feature remaining — stopping")
                break

            # Step 3a: Pairwise MI on current surviving set (exact submatrix)
            surv_in_pmi = [f for f in surviving if f in pairwise_mi.index]
            current_pmi = pairwise_mi.loc[surv_in_pmi, surv_in_pmi]

            # Step 3b: Data-driven PMI threshold
            pmi_vals = current_pmi.values[
                np.triu_indices_from(current_pmi.values, k=1)
            ]
            pmi_vals_pos = pmi_vals[pmi_vals > 0]
            if len(pmi_vals_pos) == 0:
                print(f"  [WARN] No positive pairwise MI — "
                      f"stopping (structure fully independent)")
                break
            pmi_threshold = float(np.quantile(pmi_vals_pos, pmi_quantile))

            # Step 3c: Connected components on MI > threshold
            surv_set  = [f for f in surviving if f in current_pmi.index]
            adjacency = {f: set() for f in surv_set}
            for i, fi in enumerate(surv_set):
                for j, fj in enumerate(surv_set):
                    if i >= j:
                        continue
                    if current_pmi.loc[fi, fj] > pmi_threshold:
                        adjacency[fi].add(fj)
                        adjacency[fj].add(fi)

            visited  = set()
            clusters = {}
            cid      = 0
            for feat in surv_set:
                if feat in visited:
                    continue
                cluster = []
                queue   = [feat]
                while queue:
                    f = queue.pop()
                    if f in visited:
                        continue
                    visited.add(f)
                    cluster.append(f)
                    queue.extend(adjacency[f] - visited)
                if len(cluster) > 1:
                    clusters[cid] = cluster
                    cid += 1

            in_cluster = {f for feats in clusters.values() for f in feats}

            if not in_cluster:
                print(f"  No redundant clusters at PMI threshold "
                      f"{pmi_threshold:.4f} — stopping")
                break

            print(f"  PMI threshold: {pmi_threshold:.4f} | "
                  f"{len(clusters)} clusters | "
                  f"{len(in_cluster)} features in clusters")
            for c, feats in clusters.items():
                print(f"    Cluster {c}: {feats}")

            # Step 3d: Score all features in any cluster
            scores  = {}
            k_after_map = {}
            for f in in_cluster:
                candidate = [x for x in surviving if x != f]
                if len(candidate) < 1:
                    continue
                k_after        = _kappa(candidate)
                k_after_map[f] = k_after
                scores[f]      = (kappa_now - k_after) / _pc_cost(f)

            # Step 3e: Drop single highest-scored feature
            to_drop = max(scores, key=scores.get)
            k_after = k_after_map[to_drop]
            pc_cost = _pc_cost(to_drop)
            feat_cluster = next(
                (c for c, feats in clusters.items() if to_drop in feats),
                "unknown"
            )

            kappa_path.append({
                "iteration"    : n_iterations,
                "feature"      : to_drop,
                "cluster"      : feat_cluster,
                "κ_before"     : round(kappa_now, 3),
                "κ_after"      : round(k_after, 3),
                "κ_improvement": round(kappa_now - k_after, 3),
                "drop_score"   : round(scores[to_drop], 4),
                "pc_cost"      : round(pc_cost, 4),
                "pmi_threshold": round(pmi_threshold, 4),
                "n_features_remaining": len(surviving) - 1,
            })

            surviving.remove(to_drop)
            dropped_features.append(to_drop)

            print(f"  → Dropped '{to_drop}' (cluster {feat_cluster}) | "
                  f"score={scores[to_drop]:.4f} | "
                  f"κ: {kappa_now:.2f} → {k_after:.2f} | "
                  f"pc_cost={pc_cost:.4f}")

    # ── Step 4: Post-selection ────────────────────────────────────────────
    kappa_final = _kappa(surviving)
    vif_final   = vif_scores(X_clean[surviving]) \
                  if len(surviving) > 1 else pd.Series(dtype=float)

    # Prefer the most recent PMI (computed on the actual surviving set)
    if current_pmi is not None and all(f in current_pmi.index for f in surviving):
        selected_pmi = current_pmi.loc[surviving, surviving]
    else:
        surv_in_pmi  = [f for f in surviving if f in pairwise_mi.index]
        selected_pmi = pairwise_mi.loc[surv_in_pmi, surv_in_pmi] \
                       if surv_in_pmi else pd.DataFrame()

    # ── Final report ──────────────────────────────────────────────────────
    div = "=" * 60
    print(f"\n{div}")
    print(f"  SELECTION COMPLETE")
    print(f"  Started      : {len(X_clean.columns)} features")
    print(f"  Flat dropped : {len(flat_dropped)}")
    print(f"  MI+κ dropped : {len(dropped_features) - len(flat_dropped)}")
    print(f"  Selected     : {len(surviving)}")
    print(f"  Iterations   : {n_iterations}")
    print(f"  κ baseline   : {kappa_baseline:.2f}")
    print(f"  κ final      : {kappa_final:.2f}")
    print(f"\n  Elimination path:")
    for step in kappa_path:
        bar = "█" * min(int(step["κ_after"] / 3), 20)
        print(f"    [{step['iteration']:02d}] - {step['feature']:<28} "
              f"κ: {step['κ_before']:.2f} → {step['κ_after']:.2f} "
              f"(Δ={step['κ_improvement']:.2f})  "
              f"score={step['drop_score']:.4f}  {bar}")
    print(f"\n  VIF of selected features:")
    for feat, v in vif_final.items():
        flag = "  ← HIGH" if v > 10 else ""
        print(f"    {feat:<28} VIF={v:.2f}{flag}")
    print(f"\n  Selected features: {surviving}")
    print(f"{div}\n")

    return {
        "selected_features"  : surviving,
        "dropped_features"   : dropped_features,
        "kappa_baseline"     : kappa_baseline,
        "kappa_final"        : kappa_final,
        "kappa_path"         : kappa_path,
        "vif_final"          : vif_final,
        "selected_pmi"       : selected_pmi,
        "var_floor"          : var_floor,
        "n_iterations"       : n_iterations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FEATURE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def feature_report(result: "FeatureAnalysisResult") -> pd.DataFrame:
    """
    Consolidated feature summary table combining all analyses.

    Per feature: mean pairwise MI (lower = more independent), global
    variance and rank, dominant PC and loading, correlation cluster,
    VIF (if selected), drop iteration (if eliminated by MI+κ), selected flag.

    Returns
    -------
    pd.DataFrame — one row per feature, sorted by mean_pmi ascending
    """
    if result.X is None or result.pca is None:
        raise ValueError(
            "result is incomplete — X or pca missing. "
            "Run run_full_analysis() first."
        )

    loadings  = result.pca["loadings"]
    explained = result.pca["explained"]
    var_rank  = result.variance_rank
    clusters  = result.corr_clusters
    pmi       = result.pairwise_mi
    detail    = result.selection_detail or {}
    vif       = detail.get("vif_final", pd.Series(dtype=float))
    drop_iter = {s["feature"]: s["iteration"]
                 for s in detail.get("kappa_path", [])}

    feat_cluster = {}
    if clusters:
        for cid, feats in clusters["clusters"].items():
            for f in feats:
                feat_cluster[f] = cid

    rows = []
    for feat in result.feature_cols:
        if feat in loadings.index:
            dom_pc   = loadings.loc[feat].abs().idxmax()
            dom_load = float(loadings.loc[feat, dom_pc])
            pc_var   = float(explained[dom_pc])
        else:
            dom_pc = dom_load = pc_var = np.nan

        gv  = var_rank["global_var"].get(feat, np.nan)  \
              if var_rank else np.nan
        gvr = var_rank["combined_rank"].get(feat, np.nan) \
              if var_rank else np.nan

        if pmi is not None and feat in pmi.index:
            other    = [c for c in pmi.columns if c != feat]
            mean_pmi = float(pmi.loc[feat, other].mean())
        else:
            mean_pmi = np.nan

        rows.append({
            "feature"    : feat,
            "mean_pmi"   : round(mean_pmi, 5),
            "global_var" : round(gv, 6)   if not np.isnan(gv)   else np.nan,
            "var_rank"   : int(gvr)        if not np.isnan(gvr)  else np.nan,
            "dominant_pc": dom_pc,
            "pc_loading" : round(dom_load, 4)
                           if not np.isnan(dom_load) else np.nan,
            "pc_var_expl": round(pc_var, 4)
                           if not np.isnan(pc_var)   else np.nan,
            "cluster"    : feat_cluster.get(feat, np.nan),
            "vif"        : round(float(vif[feat]), 2)
                           if feat in vif.index else np.nan,
            "dropped_iter": drop_iter.get(feat, np.nan),
            "selected"   : feat in result.selected_features,
        })

    report = pd.DataFrame(rows).set_index("feature")
    report["pmi_rank"] = report["mean_pmi"].rank(ascending=True)
    report = report.sort_values("mean_pmi")

    print(f"\n[INFO] Feature report: {len(report)} features")
    print(f"[INFO] Selected : {report['selected'].sum()} / {len(report)}")
    print(f"[INFO] Sorted by mean pairwise MI (lower = more independent)")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_analysis(df: pd.DataFrame,
                      periods: list = None,
                      std_window: int = 252,
                      corr_method: str = "pearson",
                      corr_cluster_thresh: float = 0.70,
                      variance_window: int = 252,
                      pca_n_components: int = None,
                      pca_variance_threshold: float = 0.90,
                      pairwise_mi_max_features: int = 64,
                      pairwise_mi_neighbors: int = 5,
                      pmi_quantile: float = 0.50,
                      var_floor_fraction: float = 0.10,
                      max_condition_number: float = 30.0,
                      min_pc_loading: float = 0.10,
                      pc_cost_method: str = "max_loading",
                      min_obs: int = 128) -> FeatureAnalysisResult:
    """
    Run the full feature analysis pipeline.

    Steps
    -----
    1. build_feature_matrix()   — extract numeric trend-relevant features
    2. standardize_features()   — rolling z-score (no lookahead)
    3. correlation_analysis()   — pairwise correlations + clustering
    4. variance_analysis()      — variance ranking (feeds variance floor)
    5. run_pca()                — independent trend dimensions (feeds pc_cost)
    6. pairwise_mi_matrix()     — MI between features (not vs target)
    7. select_features()        — variance floor + iterative MI-cluster/κ elimination
    8. feature_report()         — call separately on the returned result

    Selection is based on redundancy between features (pairwise MI) and
    conditioning of the selected set (κ), not on predictive MI vs r_day —
    appropriate for a description/scoring task.
    """
    if periods is None:
        periods = [21, 64, 128]

    div = "=" * 60
    print(f"\n{div}")
    print(f"  FEATURE ANALYSIS PIPELINE")
    print(f"  periods={periods}  |  std_window={std_window}")
    print(f"{div}\n")

    result            = FeatureAnalysisResult()
    result.periods    = periods
    result.std_window = std_window
    result.target_col = "pairwise_MI"

    # Step 1
    print("[STEP 1] Building feature matrix...")
    X = build_feature_matrix(df, periods, min_obs=min_obs)
    result.feature_cols = list(X.columns)
    result.X            = X

    # Step 2
    print("\n[STEP 2] Standardizing features...")
    X_std        = standardize_features(X, window=std_window)
    result.X_std = X_std
    X_std_clean  = X_std.dropna()

    # Step 3
    print("\n[STEP 3] Correlation analysis...")
    corr_out = correlation_analysis(
        X_std_clean,
        method=corr_method,
        cluster_threshold=corr_cluster_thresh
    )
    result.corr_matrix   = corr_out["corr_matrix"]
    result.corr_clusters = corr_out

    # Step 4
    print("\n[STEP 4] Variance analysis...")
    var_out = variance_analysis(X_std_clean, rolling_window=variance_window)
    result.variance_rank = var_out

    # Step 5
    print("\n[STEP 5] PCA...")
    pca_out = run_pca(
        X_std_clean,
        n_components=pca_n_components,
        variance_threshold=pca_variance_threshold
    )
    result.pca = pca_out

    # Step 6
    print("\n[STEP 6] Pairwise MI between features...")
    pmi = pairwise_mi_matrix(
        X_std_clean,
        n_neighbors=pairwise_mi_neighbors,
        max_features=pairwise_mi_max_features
    )
    result.mi_scores   = None
    result.pairwise_mi = pmi

    # Step 7
    print("\n[STEP 7] Feature selection (MI + κ)...")
    sel_out = select_features(
        X_std_clean, pca_out, var_out, pmi,
        pmi_quantile=pmi_quantile,
        var_floor_fraction=var_floor_fraction,
        max_condition_number=max_condition_number,
        min_pc_loading=min_pc_loading,
        pairwise_mi_neighbors=pairwise_mi_neighbors,
        pc_cost_method=pc_cost_method,
    )
    sel = sel_out["selected_features"]
    result.selected_features = sel
    result.selection_detail  = sel_out
    result.selected_pmi      = sel_out["selected_pmi"]
    result.selected_corr     = result.corr_matrix.loc[sel, sel]

    print(f"\n{div}")
    print(f"  FEATURE ANALYSIS COMPLETE")
    print(f"  Candidates : {len(result.feature_cols)}")
    print(f"  Selected   : {len(sel)}")
    print(f"  PCs        : {pca_out['n_components']}")
    print(f"  κ          : {sel_out['kappa_baseline']:.2f} → "
          f"{sel_out['kappa_final']:.2f}")
    print(f"  Selected   : {sel}")
    print(f"{div}\n")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from schwabttk.price_history import load_stored
    from schwabttk.price_features import compute_all_features

    # ── Load and compute features ─────────────────────────────────────────
    df  = load_stored("M2K")
    out = compute_all_features(df, periods=[21, 64, 128])

    # ── Run full feature analysis ─────────────────────────────────────────
    result = run_full_analysis(
        out,
        periods=[21, 64, 128],
        std_window=252,
        pca_variance_threshold=0.90,
        pairwise_mi_max_features=64,   # ≥ 59 candidates → no cap
        pairwise_mi_neighbors=5,
        pmi_quantile=0.50,
        var_floor_fraction=0.10,
        max_condition_number=30.0,
        min_pc_loading=0.10,
    )

    # ── Feature report ────────────────────────────────────────────────────
    report = feature_report(result)
    print("\n── Top 20 most independent features ──")
    print(report.head(20).to_string())

    # ── PCA variance breakdown ────────────────────────────────────────────
    print("\n── PCA: variance per PC ──")
    for pc, var in result.pca["explained"].items():
        cum = result.pca["cumulative"][pc]
        print(f"  {pc}: {var*100:.1f}%  (cumulative: {cum*100:.1f}%)")

    # ── Top features per PC ───────────────────────────────────────────────
    print("\n── Top features per PC (by |loading|) ──")
    for pc, feats in result.pca["feature_importance"].items():
        pct = result.pca["explained"][pc] * 100
        print(f"  {pc} ({pct:.1f}% var):")
        for feat, info in feats.items():
            print(f"    {feat:<28} loading={info['loading']:+.4f}")

    # ── Top correlated pairs ──────────────────────────────────────────────
    print("\n── Top 10 high-correlation pairs ──")
    for a, b, c in result.corr_clusters["high_corr_pairs"][:10]:
        print(f"  {a:<28} {b:<28} corr={c:+.3f}")

    # ── Correlation clusters ──────────────────────────────────────────────
    print("\n── Correlation clusters (size > 1) ──")
    for cid, feats in result.corr_clusters["clusters"].items():
        if len(feats) > 1:
            print(f"  Cluster {cid}: {feats}")

    # ── Elimination audit trail ───────────────────────────────────────────
    path = result.selection_detail["kappa_path"]
    if path:
        print("\n── MI+κ elimination path ──")
        print(pd.DataFrame(path).set_index("iteration").to_string())

    # ── Selected features ready for trend_score.py ───────────────────────
    print(f"\n── Selected features ({len(result.selected_features)}) ──")
    loadings = result.pca["loadings"]
    spmi     = result.selected_pmi
    vif      = result.selection_detail["vif_final"]
    for f in result.selected_features:
        pmi_val = (
            spmi.loc[f, [c for c in spmi.columns if c != f]].mean()
            if spmi is not None and f in spmi.index and len(spmi) > 1
            else np.nan
        )
        dom_pc  = loadings.loc[f].abs().idxmax() if f in loadings.index else "?"
        loading = float(loadings.loc[f, dom_pc]) if f in loadings.index else np.nan
        print(f"  {f:<28} mean_pmi={pmi_val:.4f}  VIF={vif.get(f, np.nan):.2f}  "
              f"dominant={dom_pc}  loading={loading:+.4f}")