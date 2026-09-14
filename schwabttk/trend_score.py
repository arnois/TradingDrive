# schwabttk/trend_score.py
"""
Trend Score
===========
Consumes selected features from feature_analysis.FeatureAnalysisResult
and produces a smooth scalar trend score ∈ [-100, +100].

This module is intentionally lean — all feature understanding (PCA,
MI, correlation, selection) lives in feature_analysis.py. This module
only handles the mechanics of turning a selected feature set into
a well-calibrated, time-varying score.

Dependency chain
----------------
    price_features.compute_all_features(df)
        → feature_analysis.run_full_analysis(df_features)
            → trend_score.compute_trend_score(df_features, result, ...)

Score construction
------------------
1. Extract selected features from df
2. Rolling z-score standardize each (no lookahead)
3. Weight each feature by its PC loading × PC variance explained
4. Normalize weights to sum to 1
5. Weighted sum → TrendScore_raw (unbounded)
6. Rolling percentile over score_window → [0, 100]
7. Rescale to [-100, +100]: TrendScore = (pct - 50) × 2

Score interpretation
--------------------
 +100 : most bullish reading in the past score_window bars
  +50 : mildly bullish
    0 : neutral
  -50 : mildly bearish
 -100 : most bearish reading in the past score_window bars
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from feature_analysis import FeatureAnalysisResult, rolling_zscore


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_weights(selected_features: list,
                    pca_result: dict) -> dict:
    """
    Compute normalized weight for each selected feature.

    Weight formula
    --------------
    w(f) = sum over PCs of [ |loading(f, PC)| × variance_explained(PC) ]

    A feature that loads strongly on a high-variance PC gets a higher weight.
    Features absent from the PCA loadings receive a baseline weight of 1.0.

    Parameters
    ----------
    selected_features : list of feature names from feature_analysis
    pca_result        : dict from feature_analysis.run_pca()

    Returns
    -------
    dict {feature: normalized_weight}  — weights sum to 1.0
    """
    loadings  = pca_result["loadings"]
    explained = pca_result["explained"]

    raw = {}
    for feat in selected_features:
        if feat in loadings.index:
            w = sum(
                abs(float(loadings.loc[feat, pc])) * float(explained[pc])
                for pc in loadings.columns
            )
        else:
            w = 1.0
        raw[feat] = w

    total   = sum(raw.values())
    weights = {k: v / total for k, v in raw.items()}
    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# TREND SCORE
# ═══════════════════════════════════════════════════════════════════════════════
def compute_trend_score(df: pd.DataFrame,
                        analysis: FeatureAnalysisResult,
                        std_window: int = 252,
                        score_window: int = 252) -> pd.DataFrame:
    """
    Compute the trend score ∈ [-100, +100] from selected features.

    Parameters
    ----------
    df           : full output of compute_all_features()
    analysis     : FeatureAnalysisResult from run_full_analysis()
    std_window   : rolling z-score window (default 252)
    score_window : rolling percentile window (default 252)

    Columns added
    -------------
    TrendScore_raw : unbounded weighted z-score combination
    TrendScore     : ∈ [-100, +100] rolling-percentile rescaled
    TrendDir       : sign(TrendScore) ∈ {-1, 0, +1}

    Returns
    -------
    pd.DataFrame — df with three new columns added
    """
    selected = analysis.selected_features
    if not selected:
        raise ValueError(
            "analysis.selected_features is empty. "
            "Run feature_analysis.run_full_analysis() first."
        )

    missing = [f for f in selected if f not in df.columns]
    if missing:
        raise ValueError(
            f"Selected features missing from df: {missing}. "
            f"Ensure df is the output of compute_all_features()."
        )

    out = df.copy()

    # Step 1-2: Extract and standardize
    X_std = pd.DataFrame(index=out.index)
    for feat in selected:
        X_std[feat] = rolling_zscore(out[feat], std_window)

    # Step 3-4: Weights
    weights = compute_weights(selected, analysis.pca)

    print("[INFO] Trend score feature weights:")
    for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
        mi  = analysis.mi_scores.get(feat, np.nan) \
              if analysis.mi_scores is not None else np.nan
        dom_pc = (
            analysis.pca["loadings"].loc[feat].abs().idxmax()
            if feat in analysis.pca["loadings"].index else "?"
        )
        print(f"  {feat:<28} weight={w:.4f}  MI={mi:.4f}  PC={dom_pc}")

    # Step 5: Weighted sum
    raw = sum(X_std[feat] * weights[feat] for feat in selected)
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
    print(f"\n[INFO] TrendScore | "
          f"min={ts.min():.1f}  max={ts.max():.1f}  "
          f"mean={ts.mean():.2f}  std={ts.std():.2f}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════════
def trend_score_pipeline(df: pd.DataFrame,
                         analysis: FeatureAnalysisResult,
                         std_window: int = 252,
                         score_window: int = 252) -> dict:
    """
    Convenience wrapper: compute trend score and return result dict.

    Parameters
    ----------
    df           : output of compute_all_features()
    analysis     : FeatureAnalysisResult from run_full_analysis()
    std_window   : rolling z-score window (default 252)
    score_window : percentile rescaling window (default 252)

    Returns
    -------
    dict:
        df_scored : df with TrendScore, TrendScore_raw, TrendDir added
        weights   : {feature: weight} used in scoring
        analysis  : the FeatureAnalysisResult passed in (for reference)
    """
    div = "=" * 60
    print(f"\n{div}")
    print(f"  TREND SCORE PIPELINE")
    print(f"  selected : {analysis.selected_features}")
    print(f"  std_window={std_window}  |  score_window={score_window}")
    print(f"{div}\n")

    df_scored = compute_trend_score(
        df, analysis,
        std_window=std_window,
        score_window=score_window
    )
    weights = compute_weights(analysis.selected_features, analysis.pca)

    print(f"\n{div}")
    print(f"  TREND SCORE COMPLETE")
    print(f"  Score range : "
          f"{df_scored['TrendScore'].min():.1f} to "
          f"{df_scored['TrendScore'].max():.1f}")
    print(f"{div}\n")

    return {
        "df_scored": df_scored,
        "weights"  : weights,
        "analysis" : analysis,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import plotly.graph_objects as go
    from schwabttk.price_history import load_stored
    from schwabttk.price_features import compute_all_features
    from schwabttk.feature_analysis import run_full_analysis
    from schwabttk.visualize import show

    # Step 1: Features
    df  = load_stored("M2K")
    out = compute_all_features(df, periods=[21, 64, 128])

    # Step 2: Feature analysis
    analysis = run_full_analysis(
        out,
        periods=[21, 64, 128],
        std_window=252,
        pca_variance_threshold=0.90,
        mi_target_col="R_21",
        n_per_pc=2,
        max_corr=0.70,
        mi_quantile=0.25
    )

    # Step 3: Trend score
    result    = trend_score_pipeline(out, analysis)
    df_scored = result["df_scored"]

    print("\n── Last 10 bars ──")
    print(df_scored[["datetime", "TrendScore", "TrendScore_raw",
                      "TrendDir", "Regime"]].tail(10).to_string(index=False))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"].clip(lower=0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(38,166,154,0.20)", name="Bull"
    ))
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"].clip(upper=0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(239,83,80,0.20)", name="Bear"
    ))
    fig.add_trace(go.Scatter(
        x=df_scored["datetime"],
        y=df_scored["TrendScore"],
        mode="lines", name="TrendScore",
        line=dict(color="#D1D4DC", width=1.2)
    ))
    for y_val, color in [(50, "#ef5350"), (-50, "#26a69a"), (0, "white")]:
        fig.add_hline(y=y_val, line_dash="dash",
                      line_color=color, opacity=0.4)
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