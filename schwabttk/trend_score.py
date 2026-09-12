# schwabttk/trend_score.py
"""
Trend Score
===========
Stateful pipeline that consumes the output of price_features.compute_all_features()
and produces a smooth scalar trend score ∈ [-100, +100].

Unlike price_features.py (deterministic, stateless), this module requires
a fitting phase: PCA must be fitted on historical data before scores
can be computed. Same input does NOT always give the same output —
the score depends on the fitting window.

Pipeline
--------
1. build_feature_matrix()  — select numeric, signed, trend-relevant features
2. standardize_features()  — rolling z-score (no lookahead)
3. run_pca()               — discover independent trend dimensions
4. select_features()       — most informative + least redundant per PC
5. compute_trend_score()   — weighted combination → rolling percentile → [-100,+100]

Full pipeline via trend_score_pipeline() in one call.
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def _trend_feature_cols(df: pd.DataFrame,
                        periods: list) -> list:
    """
    Return list of numeric trend-relevant feature columns present in df.

    Inclusion criteria
    ------------------
    - Continuous numeric (not categorical: Pattern_*, Regime, is_* flags)
    - Signed (carries directional information)
    - Not a raw price level (EMA excluded, EMA_slope included)
    - Not a volume level (dVol, volume excluded; RVOL, VTC, VWR included)

    Per-bar (period-independent)
    ----------------------------
    r_day, CLV, BodyRatio, UpperRatio, LowerRatio, Conviction

    Per-period (for each p in periods)
    ------------------------------------
    R, TM, ER, DC, AC1, AC5, AvgRunPos, AvgRunNeg, DomRun,
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
        print(f"[WARN] {len(missing)} candidate columns not found in df "
              f"(run compute_all_features first): {missing}")
    return present


def build_feature_matrix(df: pd.DataFrame,
                         periods: list = None,
                         min_obs: int = 128) -> pd.DataFrame:
    """
    Extract and clean the numeric feature matrix for trend scoring.

    Parameters
    ----------
    df      : output of compute_all_features()
    periods : must match what was used in compute_all_features()
    min_obs : minimum valid observations required per column

    Returns
    -------
    pd.DataFrame of numeric features with NaN rows dropped
    """
    if periods is None:
        periods = [21, 64, 128]

    cols = _trend_feature_cols(df, periods)
    X    = df[cols].copy()

    # Drop columns with insufficient valid observations
    valid_cols = [c for c in X.columns
                  if X[c].notna().sum() >= min_obs]
    dropped = set(cols) - set(valid_cols)
    if dropped:
        print(f"[INFO] Dropped {len(dropped)} columns "
              f"with < {min_obs} valid observations: {sorted(dropped)}")

    X = X[valid_cols].dropna()
    print(f"[INFO] Feature matrix: {X.shape[0]} rows × {X.shape[1]} cols")
    return X


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score standardization — no lookahead bias.
    Uses only past observations within each window.
    Returns NaN for the first `window` observations.
    """
    mu  = series.rolling(window).mean()
    sig = series.rolling(window).std().replace(0, np.nan)
    return (series - mu) / sig


def standardize_features(X: pd.DataFrame,
                         window: int = 252) -> pd.DataFrame:
    """
    Rolling z-score each feature column independently.

    Parameters
    ----------
    X      : raw feature matrix from build_feature_matrix()
    window : rolling window for mean/std (default 252 ≈ 1 trading year)

    Returns
    -------
    pd.DataFrame of standardized features (same shape as X)
    """
    return X.apply(lambda col: rolling_zscore(col, window))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PCA
# ═══════════════════════════════════════════════════════════════════════════════
def run_pca(X_std: pd.DataFrame,
            n_components: int = None,
            variance_threshold: float = 0.90) -> dict:
    """
    Fit PCA on the standardized feature matrix to discover independent
    trend dimensions.

    Parameters
    ----------
    X_std              : standardized feature matrix (NaNs dropped)
    n_components       : fixed number of PCs (None = auto from threshold)
    variance_threshold : if n_components is None, keep enough PCs to
                         explain this fraction of total variance (0.90)

    Returns
    -------
    dict:
        pca               : fitted sklearn PCA object
        loadings          : pd.DataFrame (features × PCs) — PC directions
        explained         : pd.Series — variance explained per PC
        cumulative        : pd.Series — cumulative variance explained
        n_components      : int — number of PCs selected
        scores            : pd.DataFrame — PC scores (same index as X_std)
        feature_importance: dict {PC: {feature: {loading, abs_loading}}}
                            top-5 features per PC by absolute loading
    """
    X_clean = X_std.dropna()

    if n_components is None:
        pca_full = PCA().fit(X_clean)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumvar >= variance_threshold) + 1)
        print(f"[INFO] Auto-selected {n_components} PCs → "
              f"{cumvar[n_components-1]*100:.1f}% variance explained")

    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(X_clean)

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

    # Top-5 features per PC by absolute loading
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

    scores_df = pd.DataFrame(scores, index=X_clean.index, columns=pc_labels)

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
# STEP 4 — FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
def select_features(X_std: pd.DataFrame,
                    pca_result: dict,
                    target_col: str = "R_21",
                    n_per_pc: int = 2,
                    max_corr: float = 0.70,
                    mi_quantile: float = 0.25) -> dict:
    """
    Select the most informative and least redundant features per PC.

    Selection method (per PC)
    -------------------------
    1. Rank features by absolute loading on that PC
    2. Filter by mutual information with target_col > mi_quantile threshold
       (MI measures non-linear dependence on forward returns)
    3. Among survivors, reject if pairwise |corr| > max_corr with
       already-selected features (redundancy filter)
    4. Keep up to n_per_pc features per PC

    Features selected this way are:
    - Directionally aligned with independent trend dimensions (PC loadings)
    - Predictive of forward returns (MI filter)
    - Not redundant with each other (correlation filter)
    → ideal for stable covariance matrices downstream

    Parameters
    ----------
    X_std       : standardized feature matrix (NaNs dropped)
    pca_result  : output of run_pca()
    target_col  : MI target column (default "R_21" — 1-month forward return)
    n_per_pc    : max features to select per PC (default 2)
    max_corr    : maximum pairwise |correlation| allowed (default 0.70)
    mi_quantile : MI score quantile threshold for informativeness (0.25)

    Returns
    -------
    dict:
        selected_features : list of selected feature names (ordered)
        per_pc            : dict {PC: [feature names]}
        corr_matrix       : correlation matrix of selected features
        mi_scores         : pd.Series of mutual information scores
    """
    loadings = pca_result["loadings"]
    X_clean  = X_std.dropna()

    # Mutual information against forward return (target shifted -1)
    if target_col in X_clean.columns:
        target = X_clean[target_col].shift(-1).dropna()
        X_mi   = X_clean.loc[target.index].fillna(0)
        mi_raw = mutual_info_regression(X_mi, target, random_state=42)
        mi_scores = pd.Series(mi_raw, index=X_mi.columns).sort_values(
            ascending=False
        )
        mi_thresh = mi_scores.quantile(mi_quantile)
    else:
        print(f"[WARN] target_col '{target_col}' not in X_std — "
              f"falling back to PC1 absolute loading for ranking")
        mi_scores = loadings["PC1"].abs()
        mi_thresh = mi_scores.quantile(mi_quantile)

    selected = []
    per_pc   = {}

    for pc in loadings.columns:
        pc_rank = loadings[pc].abs().sort_values(ascending=False)
        chosen  = []

        for feat in pc_rank.index:
            if feat in selected:
                continue

            # MI threshold — must be informative
            if mi_scores.get(feat, 0) < mi_thresh:
                continue

            # Redundancy check — must not be correlated with chosen/selected
            redundant = False
            for existing in (chosen + selected):
                if existing in X_clean.columns and feat in X_clean.columns:
                    if abs(X_clean[feat].corr(X_clean[existing])) > max_corr:
                        redundant = True
                        break
            if redundant:
                continue

            chosen.append(feat)
            if len(chosen) >= n_per_pc:
                break

        per_pc[pc] = chosen
        selected  += [f for f in chosen if f not in selected]

    corr_matrix = (X_clean[selected].corr()
                   if selected else pd.DataFrame())

    print(f"[INFO] Selected {len(selected)} features across "
          f"{len(loadings.columns)} PCs:")
    for pc, feats in per_pc.items():
        pct = pca_result["explained"][pc] * 100
        print(f"  {pc} ({pct:.1f}% var explained): {feats}")

    return {
        "selected_features": selected,
        "per_pc"           : per_pc,
        "corr_matrix"      : corr_matrix,
        "mi_scores"        : mi_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TREND SCORE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_trend_score(df: pd.DataFrame,
                        selected_features: list,
                        pca_result: dict,
                        score_window: int = 252,
                        std_window: int = 252) -> pd.DataFrame:
    """
    Compute the final trend score ∈ [-100, +100].

    Method
    ------
    1. Extract selected features from df
    2. Rolling z-score each feature (no lookahead)
    3. Compute weight per feature:
         w(f) = sum over PCs of [ |loading(f, PC)| × variance_explained(PC) ]
       → features driving high-variance PCs get higher weight
    4. Normalize weights to sum to 1
    5. Weighted sum of z-scores → TrendScore_raw (unbounded)
    6. Rolling percentile of raw score over score_window → [0, 100]
    7. Rescale to [-100, +100]: TrendScore = (pct - 50) × 2

    Columns added to df
    -------------------
    TrendScore_raw : unnormalized weighted combination (unbounded)
    TrendScore     : ∈ [-100, +100], rolling percentile rescaled
    TrendDir       : sign(TrendScore) ∈ {-1, 0, +1}

    Parameters
    ----------
    df                : output of compute_all_features() (full df, not just X)
    selected_features : list from select_features()
    pca_result        : output of run_pca()
    score_window      : rolling window for percentile rescaling (252)
    std_window        : rolling window for z-score standardization (252)
    """
    out      = df.copy()
    loadings = pca_result["loadings"]
    explained = pca_result["explained"]

    # Step 1-2: Extract and standardize selected features
    feats = [f for f in selected_features if f in out.columns]
    if not feats:
        raise ValueError("None of the selected features are present in df.")

    X_std = pd.DataFrame(index=out.index)
    for feat in feats:
        X_std[feat] = rolling_zscore(out[feat], std_window)

    # Step 3-4: Compute and normalize weights
    weights = {}
    for feat in feats:
        if feat in loadings.index:
            w = sum(
                abs(float(loadings.loc[feat, pc])) * float(explained[pc])
                for pc in loadings.columns
            )
        else:
            w = 1.0   # fallback for features not in loadings
        weights[feat] = w

    total   = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    print("[INFO] Trend score feature weights:")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {feat:<28} {w:.4f}")

    # Step 5: Weighted sum → raw score
    raw = sum(X_std[feat] * weights[feat] for feat in feats)
    out["TrendScore_raw"] = raw

    # Step 6-7: Rolling percentile → [-100, +100]
    out["TrendScore"] = (
        raw
        .rolling(score_window)
        .apply(
            lambda x: percentileofscore(x, x.iloc[-1], kind="rank"),
            raw=False
        )
        .sub(50)
        .mul(2)
    )

    out["TrendDir"] = np.sign(out["TrendScore"])

    ts = out["TrendScore"].dropna()
    print(f"[INFO] TrendScore | "
          f"min={ts.min():.1f}  max={ts.max():.1f}  "
          f"mean={ts.mean():.2f}  std={ts.std():.2f}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def trend_score_pipeline(df: pd.DataFrame,
                         periods: list = None,
                         n_components: int = None,
                         variance_threshold: float = 0.90,
                         n_per_pc: int = 2,
                         max_corr: float = 0.70,
                         mi_quantile: float = 0.25,
                         std_window: int = 252,
                         score_window: int = 252,
                         target_col: str = "R_21",
                         min_obs: int = 128) -> dict:
    """
    Full trend score pipeline from feature-engineered DataFrame.

    Steps
    -----
    1. build_feature_matrix()  — select numeric trend-relevant features
    2. standardize_features()  — rolling z-score (no lookahead)
    3. run_pca()               — discover independent trend dimensions
    4. select_features()       — informative + non-redundant per PC
    5. compute_trend_score()   — weighted score → [-100, +100]

    Parameters
    ----------
    df                 : output of compute_all_features()
    periods            : must match compute_all_features() (default [21,64,128])
    n_components       : PCA components (None = auto from variance_threshold)
    variance_threshold : cumulative variance target for auto PCA (0.90)
    n_per_pc           : features to select per PC (default 2)
    max_corr           : max pairwise |corr| for redundancy filter (0.70)
    mi_quantile        : MI quantile threshold for informativeness (0.25)
    std_window         : rolling window for z-score standardization (252)
    score_window       : rolling window for percentile rescaling (252)
    target_col         : MI target column for feature selection ("R_21")
    min_obs            : min valid obs per column in feature matrix (128)

    Returns
    -------
    dict:
        df_scored   : df with TrendScore, TrendScore_raw, TrendDir added
        pca_result  : full PCA output (loadings, explained, scores, etc.)
        selection   : feature selection output (selected_features, per_pc,
                      corr_matrix, mi_scores)
        feature_cols: all candidate feature columns used
        weights     : {feature: weight} dict used in scoring
    """
    if periods is None:
        periods = [21, 64, 128]

    div = "=" * 60
    print(f"\n{div}")
    print(f"  TREND SCORE PIPELINE")
    print(f"  periods={periods}  |  std_window={std_window}  "
          f"|  score_window={score_window}")
    print(f"{div}\n")

    # ── Step 1 ────────────────────────────────────────────────────────────
    print("[STEP 1] Building feature matrix...")
    X = build_feature_matrix(df, periods, min_obs=min_obs)

    # ── Step 2 ────────────────────────────────────────────────────────────
    print("\n[STEP 2] Rolling z-score standardization...")
    X_std = standardize_features(X, window=std_window)

    # ── Step 3 ────────────────────────────────────────────────────────────
    print("\n[STEP 3] PCA...")
    X_std_clean = X_std.dropna()
    pca_result  = run_pca(X_std_clean, n_components, variance_threshold)

    print(f"\n  Variance explained per PC:")
    for pc in pca_result["explained"].index:
        var = pca_result["explained"][pc]
        cum = pca_result["cumulative"][pc]
        print(f"    {pc}: {var*100:.1f}%  (cumulative: {cum*100:.1f}%)")

    print(f"\n  Top features per PC (by |loading|):")
    for pc, feats in pca_result["feature_importance"].items():
        print(f"    {pc}:")
        for feat, info in feats.items():
            print(f"      {feat:<28} loading={info['loading']:+.4f}")

    # ── Step 4 ────────────────────────────────────────────────────────────
    print("\n[STEP 4] Feature selection...")
    selection = select_features(
        X_std_clean, pca_result,
        target_col=target_col,
        n_per_pc=n_per_pc,
        max_corr=max_corr,
        mi_quantile=mi_quantile
    )

    print(f"\n  Selected feature pairwise correlations:")
    print(selection["corr_matrix"].round(3).to_string())

    # ── Step 5 ────────────────────────────────────────────────────────────
    print("\n[STEP 5] Computing trend score...")
    df_scored = compute_trend_score(
        df,
        selection["selected_features"],
        pca_result,
        score_window=score_window,
        std_window=std_window
    )

    # Extract weights from df_scored computation for return
    loadings  = pca_result["loadings"]
    explained = pca_result["explained"]
    feats     = [f for f in selection["selected_features"]
                 if f in df_scored.columns]
    raw_w = {
        feat: sum(
            abs(float(loadings.loc[feat, pc])) * float(explained[pc])
            for pc in loadings.columns
        ) if feat in loadings.index else 1.0
        for feat in feats
    }
    total = sum(raw_w.values())
    weights = {k: round(v / total, 6) for k, v in raw_w.items()}

    print(f"\n{div}")
    print(f"  TREND SCORE PIPELINE COMPLETE")
    print(f"  Features used : {len(feats)}")
    print(f"  Score range   : "
          f"{df_scored['TrendScore'].min():.1f} to "
          f"{df_scored['TrendScore'].max():.1f}")
    print(f"{div}\n")

    return {
        "df_scored"   : df_scored,
        "pca_result"  : pca_result,
        "selection"   : selection,
        "feature_cols": list(X.columns),
        "weights"     : weights,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import plotly.graph_objects as go
    from schwabttk.price_history import load_stored
    from schwabttk.price_features import compute_all_features
    from schwabttk.visualize import show

    # ── Load and compute features ─────────────────────────────────────────
    df  = load_stored("M2K")
    out = compute_all_features(df, periods=[21, 64, 128])

    # ── Run trend score pipeline ──────────────────────────────────────────
    result    = trend_score_pipeline(
        out,
        periods=[21, 64, 128],
        variance_threshold=0.90,
        n_per_pc=2,
        std_window=252,
        score_window=252,
        target_col="R_21"
    )
    df_scored = result["df_scored"]

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n── Last 10 bars ──")
    print(df_scored[["datetime", "TrendScore", "TrendScore_raw",
                      "TrendDir", "Regime"]].tail(10).to_string(index=False))

    print("\n── PCA feature importance ──")
    for pc, feats in result["pca_result"]["feature_importance"].items():
        pct = result["pca_result"]["explained"][pc] * 100
        print(f"\n{pc} ({pct:.1f}% variance):")
        for feat, info in feats.items():
            print(f"  {feat:<28} {info['loading']:+.4f}")

    print("\n── Selected features and weights ──")
    for feat, w in sorted(result["weights"].items(), key=lambda x: -x[1]):
        print(f"  {feat:<28} {w:.4f}")

    # ── Plot TrendScore ───────────────────────────────────────────────────
    fig = go.Figure()

    # Shade positive/negative regions
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"].clip(lower=0),
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(38,166,154,0.25)",
        name="Bull"
    ))
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"].clip(upper=0),
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(239,83,80,0.25)",
        name="Bear"
    ))

    # Score line
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"],
        mode="lines",
        name="TrendScore",
        line=dict(color="#D1D4DC", width=1.2)
    ))

    # Reference lines
    for y, color in [(50, "#ef5350"), (-50, "#26a69a"), (0, "white")]:
        fig.add_hline(
            y=y,
            line_dash="dash",
            line_color=color,
            opacity=0.4
        )

    fig.update_layout(
        title=dict(text="M2K — Trend Score", font=dict(size=18)),
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        height=450,
        margin=dict(l=60, r=50, t=60, b=40),
        yaxis=dict(title="Score", range=[-100, 100],
                   gridcolor="#2A2E39"),
        xaxis=dict(title="Date", gridcolor="#2A2E39"),
        hovermode="x unified",
        showlegend=False
    )

    show(fig, "M2K_trend_score")