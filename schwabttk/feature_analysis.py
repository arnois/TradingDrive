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
6.  mutual_information()      — which features predict forward returns
7.  select_features()         — combine PCA + MI + redundancy filter
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
    """
    Container for all feature analysis outputs.
    Pass .selected_features and .pca directly to trend_score.py.

    Attributes
    ----------
    feature_cols      : all candidate feature columns
    X                 : raw feature matrix (pd.DataFrame)
    X_std             : standardized feature matrix (pd.DataFrame)
    corr_matrix       : full pairwise correlation matrix
    corr_clusters     : dict {cluster_id: [feature names]}
    variance_rank     : pd.Series — features ranked by rolling variance
    pca               : dict — full PCA output (see run_pca docstring)
    mi_scores         : pd.Series — mutual information scores vs target
    selected_features : list — final selected feature names
    selection_detail  : dict {PC: [features]} — which features per PC
    selected_corr     : correlation matrix of selected features only
    periods           : periods used in price_features pipeline
    std_window        : rolling window used for standardization
    target_col        : MI target column used
    """
    feature_cols      : list = field(default_factory=list)
    X                 : Optional[pd.DataFrame] = None
    X_std             : Optional[pd.DataFrame] = None
    corr_matrix       : Optional[pd.DataFrame] = None
    corr_clusters     : Optional[dict] = None
    variance_rank     : Optional[pd.Series] = None
    pca               : Optional[dict] = None
    mi_scores         : Optional[pd.Series] = None
    selected_features : list = field(default_factory=list)
    selection_detail  : Optional[dict] = None
    selected_corr     : Optional[pd.DataFrame] = None
    periods           : list = field(default_factory=lambda: [21, 64, 128])
    std_window        : int = 252
    target_col        : str = "R_21"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def _candidate_feature_cols(df: pd.DataFrame,
                             periods: list) -> list:
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
    r_day, CLV, BodyRatio, UpperRatio, LowerRatio, Conviction

    Per-period rolling features
    ---------------------------
    R, TM, ER, DC, AC1, AC5, AC21,
    AvgRunPos, AvgRunNeg, DomRun,
    RVOL, VTC, VWR, PVE, PVE_pct, EMA_slope, MOM
    """
    per_bar = [
        "r_day",
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
            f"R{sfx}",
            f"TM{sfx}",
            f"ER{sfx}",
            f"DC{sfx}",
            f"AC1{sfx}",
            f"AC5{sfx}",
            f"AC21{sfx}",
            f"AvgRunPos{sfx}",
            f"AvgRunNeg{sfx}",
            f"DomRun{sfx}",
            f"RVOL{sfx}",
            f"VTC{sfx}",
            f"VWR{sfx}",
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
def variance_analysis(X: pd.DataFrame,
                      rolling_window: int = 252) -> dict:
    """
    Rank features by the amount of variance they carry.

    Two complementary measures:
    - Global variance  : var(feature) across all observations
    - Rolling variance : mean of rolling(window).var() — how much
                         the feature changes over time (time-varying signal)

    Features with high variance carry more information but may also
    be noisier. Use in combination with MI scores for final selection.

    Parameters
    ----------
    X              : feature matrix (raw, before standardization)
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
                            each column is a PC direction in feature space
                            large |loading| → feature strongly represents that PC
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
        index=X_std.columns,
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
# STEP 6 — MUTUAL INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
def mutual_information(X_std: pd.DataFrame,
                        target_col: str = "R_21",
                        forward_shift: int = 1,
                        n_neighbors: int = 5) -> pd.Series:
    """
    Compute mutual information between each feature and a target.

    MI measures non-linear dependence — it captures relationships
    that correlation misses. Higher MI → feature more predictive
    of future returns.

    Parameters
    ----------
    X_std         : standardized feature matrix (NaNs dropped)
    target_col    : column to use as prediction target (default "R_21")
    forward_shift : how many bars ahead for the target (default 1)
    n_neighbors   : k for k-NN MI estimator (default 5)

    Returns
    -------
    pd.Series — MI score per feature (sorted descending)
                Higher = more predictive of the target
    """
    X_clean = X_std.dropna()

    if target_col not in X_clean.columns:
        raise ValueError(
            f"target_col '{target_col}' not in feature matrix. "
            f"Available: {list(X_clean.columns)}"
        )

    target  = X_clean[target_col].shift(-forward_shift).dropna()
    X_mi    = X_clean.loc[target.index].fillna(0)

    mi_raw  = mutual_info_regression(
        X_mi, target,
        n_neighbors=n_neighbors,
        random_state=42
    )
    mi_scores = pd.Series(mi_raw, index=X_mi.columns).sort_values(
        ascending=False
    )

    print(f"[INFO] Mutual information | target={target_col} "
          f"shift={forward_shift} | top-10:")
    for feat, score in mi_scores.head(10).items():
        print(f"  {feat:<28} MI={score:.4f}")

    return mi_scores


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def select_features(X_std: pd.DataFrame,
                    pca_result: dict,
                    mi_scores: pd.Series,
                    n_per_pc: int = 2,
                    max_corr: float = 0.70,
                    mi_quantile: float = 0.25) -> dict:
    """
    Select the most informative and least redundant features per PC.

    Selection logic (per PC, in order)
    -----------------------------------
    1. Rank features by absolute loading on this PC
    2. Filter: MI score must exceed mi_quantile threshold
    3. Filter: pairwise |corr| with already-selected features < max_corr
    4. Keep up to n_per_pc features per PC

    Features passing all three filters are:
    - Aligned with independent trend dimensions   (PC loadings)
    - Predictive of forward returns               (MI filter)
    - Not redundant with each other               (correlation filter)
    → stable covariance matrices downstream

    Parameters
    ----------
    X_std       : standardized feature matrix (NaNs dropped)
    pca_result  : output of run_pca()
    mi_scores   : output of mutual_information()
    n_per_pc    : max features to select per PC (default 2)
    max_corr    : max pairwise |corr| allowed between selected (0.70)
    mi_quantile : MI quantile threshold for informativeness (0.25)

    Returns
    -------
    dict:
        selected_features : list — final selected feature names
        per_pc            : dict {PC: [feature names]}
        selected_corr     : correlation matrix of selected features
        rejection_log     : list of (feature, PC, reason) for rejected features
    """
    loadings  = pca_result["loadings"]
    X_clean   = X_std.dropna()
    mi_thresh = mi_scores.quantile(mi_quantile)

    selected     = []
    per_pc       = {}
    rejection_log = []

    for pc in loadings.columns:
        pc_rank = loadings[pc].abs().sort_values(ascending=False)
        chosen  = []

        for feat in pc_rank.index:
            if feat in selected:
                rejection_log.append((feat, pc, "already selected"))
                continue

            # MI threshold
            mi_val = mi_scores.get(feat, 0)
            if mi_val < mi_thresh:
                rejection_log.append(
                    (feat, pc, f"MI={mi_val:.4f} < threshold={mi_thresh:.4f}")
                )
                continue

            # Redundancy check
            redundant_with = None
            for existing in (chosen + selected):
                if existing in X_clean.columns and feat in X_clean.columns:
                    c = abs(X_clean[feat].corr(X_clean[existing]))
                    if c > max_corr:
                        redundant_with = existing
                        break
            if redundant_with:
                rejection_log.append(
                    (feat, pc,
                     f"|corr|={c:.3f} > {max_corr} with '{redundant_with}'")
                )
                continue

            chosen.append(feat)
            if len(chosen) >= n_per_pc:
                break

        per_pc[pc] = chosen
        selected  += [f for f in chosen if f not in selected]

    selected_corr = (
        X_clean[selected].corr() if selected else pd.DataFrame()
    )

    print(f"\n[INFO] Feature selection complete | "
          f"{len(selected)} features selected:")
    for pc, feats in per_pc.items():
        pct = pca_result["explained"][pc] * 100
        print(f"  {pc} ({pct:.1f}% var): {feats}")

    print(f"\n[INFO] Selected feature pairwise correlations:")
    print(selected_corr.round(3).to_string())

    return {
        "selected_features": selected,
        "per_pc"           : per_pc,
        "selected_corr"    : selected_corr,
        "rejection_log"    : rejection_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FEATURE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def feature_report(result: "FeatureAnalysisResult") -> pd.DataFrame:
    """
    Produce a consolidated feature summary table combining all analyses.

    For each feature in the candidate set, computes:
    - Global variance rank
    - Rolling variance rank
    - MI score and MI rank
    - Dominant PC (PC with highest |loading|) and that loading value
    - Correlation cluster membership
    - Whether the feature was selected

    Parameters
    ----------
    result : FeatureAnalysisResult from run_full_analysis()

    Returns
    -------
    pd.DataFrame — one row per feature, all metrics, sorted by MI rank
    """
    if result.X is None or result.pca is None or result.mi_scores is None:
        raise ValueError(
            "result is incomplete. Run run_full_analysis() first."
        )

    loadings   = result.pca["loadings"]
    explained  = result.pca["explained"]
    mi         = result.mi_scores
    var_rank   = result.variance_rank
    clusters   = result.corr_clusters

    # Invert clusters: feature → cluster_id
    feat_cluster = {}
    if clusters:
        for cid, feats in clusters["clusters"].items():
            for f in feats:
                feat_cluster[f] = cid

    rows = []
    for feat in result.feature_cols:
        # Dominant PC by absolute loading
        if feat in loadings.index:
            dom_pc  = loadings.loc[feat].abs().idxmax()
            dom_load = float(loadings.loc[feat, dom_pc])
            pc_var  = float(explained[dom_pc])
        else:
            dom_pc = dom_load = pc_var = np.nan

        # Global variance rank (1 = highest variance)
        gv  = var_rank["global_var"].get(feat, np.nan)
        gvr = var_rank["combined_rank"].get(feat, np.nan)

        rows.append({
            "feature"      : feat,
            "mi_score"     : round(mi.get(feat, np.nan), 5),
            "mi_rank"      : int(mi.rank(ascending=False).get(feat, np.nan))
                             if feat in mi.index else np.nan,
            "global_var"   : round(gv, 6) if not np.isnan(gv) else np.nan,
            "var_rank"     : int(gvr) if not np.isnan(gvr) else np.nan,
            "dominant_pc"  : dom_pc,
            "pc_loading"   : round(dom_load, 4)
                             if not np.isnan(dom_load) else np.nan,
            "pc_var_expl"  : round(pc_var, 4)
                             if not np.isnan(pc_var) else np.nan,
            "cluster"      : feat_cluster.get(feat, np.nan),
            "selected"     : feat in result.selected_features,
        })

    report = (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("mi_rank")
    )

    print(f"\n[INFO] Feature report: {len(report)} features")
    print(f"[INFO] Selected: {report['selected'].sum()} / {len(report)}")
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
                      mi_target_col: str = "R_21",
                      mi_forward_shift: int = 1,
                      mi_n_neighbors: int = 5,
                      n_per_pc: int = 2,
                      max_corr: float = 0.70,
                      mi_quantile: float = 0.25,
                      min_obs: int = 128) -> FeatureAnalysisResult:
    """
    Run the full feature analysis pipeline and return a FeatureAnalysisResult.

    Steps
    -----
    1. build_feature_matrix()   — extract numeric trend-relevant features
    2. standardize_features()   — rolling z-score (no lookahead)
    3. correlation_analysis()   — pairwise correlations + clustering
    4. variance_analysis()      — rank features by variance
    5. run_pca()                — discover independent trend dimensions
    6. mutual_information()     — rank features by predictive power
    7. select_features()        — informative + independent subset
    8. (report available via feature_report(result))

    Parameters
    ----------
    df                    : output of compute_all_features()
    periods               : must match compute_all_features() [21,64,128]
    std_window            : rolling z-score window (252)
    corr_method           : 'pearson' or 'spearman' for correlation (pearson)
    corr_cluster_thresh   : |corr| threshold for clustering (0.70)
    variance_window       : rolling window for variance analysis (252)
    pca_n_components      : fixed PCA components (None = auto)
    pca_variance_threshold: cumulative variance for auto PCA (0.90)
    mi_target_col         : MI target column (default 'R_21')
    mi_forward_shift      : forward shift for MI target (1)
    mi_n_neighbors        : k-NN for MI estimator (5)
    n_per_pc              : features to select per PC (2)
    max_corr              : max pairwise |corr| in selection (0.70)
    mi_quantile           : MI quantile threshold in selection (0.25)
    min_obs               : min valid obs per column (128)

    Returns
    -------
    FeatureAnalysisResult — all outputs bundled, ready for trend_score.py
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
    result.target_col = mi_target_col

    # ── Step 1 ────────────────────────────────────────────────────────────
    print("[STEP 1] Building feature matrix...")
    X = build_feature_matrix(df, periods, min_obs=min_obs)
    result.feature_cols = list(X.columns)
    result.X            = X

    # ── Step 2 ────────────────────────────────────────────────────────────
    print("\n[STEP 2] Standardizing features...")
    X_std       = standardize_features(X, window=std_window)
    result.X_std = X_std

    X_std_clean = X_std.dropna()

    # ── Step 3 ────────────────────────────────────────────────────────────
    print("\n[STEP 3] Correlation analysis...")
    corr_out             = correlation_analysis(
        X_std_clean,
        method=corr_method,
        cluster_threshold=corr_cluster_thresh
    )
    result.corr_matrix   = corr_out["corr_matrix"]
    result.corr_clusters = corr_out

    # ── Step 4 ────────────────────────────────────────────────────────────
    print("\n[STEP 4] Variance analysis...")
    var_out            = variance_analysis(X_std_clean,
                                           rolling_window=variance_window)
    result.variance_rank = var_out

    # ── Step 5 ────────────────────────────────────────────────────────────
    print("\n[STEP 5] PCA...")
    pca_out    = run_pca(
        X_std_clean,
        n_components=pca_n_components,
        variance_threshold=pca_variance_threshold
    )
    result.pca = pca_out

    # ── Step 6 ────────────────────────────────────────────────────────────
    print("\n[STEP 6] Mutual information...")
    mi_scores        = mutual_information(
        X_std_clean,
        target_col=mi_target_col,
        forward_shift=mi_forward_shift,
        n_neighbors=mi_n_neighbors
    )
    result.mi_scores = mi_scores

    # ── Step 7 ────────────────────────────────────────────────────────────
    print("\n[STEP 7] Feature selection...")
    sel_out = select_features(
        X_std_clean, pca_out, mi_scores,
        n_per_pc=n_per_pc,
        max_corr=max_corr,
        mi_quantile=mi_quantile
    )
    result.selected_features = sel_out["selected_features"]
    result.selection_detail  = sel_out["per_pc"]
    result.selected_corr     = sel_out["selected_corr"]

    print(f"\n{div}")
    print(f"  FEATURE ANALYSIS COMPLETE")
    print(f"  Candidates : {len(result.feature_cols)}")
    print(f"  Selected   : {len(result.selected_features)}")
    print(f"  PCs        : {pca_out['n_components']}")
    print(f"  Selected features: {result.selected_features}")
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
        mi_target_col="R_21",
        n_per_pc=2,
        max_corr=0.70,
        mi_quantile=0.25
    )

    # ── Feature report ────────────────────────────────────────────────────
    report = feature_report(result)
    print("\n── Top 20 features by MI rank ──")
    print(report.head(20).to_string())

    # ── PCA variance breakdown ────────────────────────────────────────────
    print("\n── PCA: variance per PC ──")
    for pc, var in result.pca["explained"].items():
        cum = result.pca["cumulative"][pc]
        print(f"  {pc}: {var*100:.1f}%  (cumulative: {cum*100:.1f}%)")

    # ── Top correlated pairs ──────────────────────────────────────────────
    print("\n── Top 10 high-correlation pairs ──")
    for a, b, c in result.corr_clusters["high_corr_pairs"][:10]:
        print(f"  {a:<28} {b:<28} corr={c:+.3f}")

    # ── Correlation clusters ──────────────────────────────────────────────
    print("\n── Correlation clusters ──")
    for cid, feats in result.corr_clusters["clusters"].items():
        if len(feats) > 1:
            print(f"  Cluster {cid}: {feats}")

    # ── Selected features ready for trend_score.py ───────────────────────
    print(f"\n── Selected features ({len(result.selected_features)}) ──")
    for f in result.selected_features:
        mi  = result.mi_scores.get(f, np.nan)
        pc  = result.pca["loadings"].loc[f].abs().idxmax() \
              if f in result.pca["loadings"].index else "?"
        load = result.pca["loadings"].loc[f, pc] \
               if f in result.pca["loadings"].index else np.nan
        print(f"  {f:<28} MI={mi:.4f}  dominant={pc}  loading={load:+.4f}")