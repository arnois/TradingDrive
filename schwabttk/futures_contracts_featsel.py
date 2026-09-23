#%% PRICE FEATURES
from schwabttk.price_history import load_stored
from schwabttk.price_features import compute_all_features
# import importlib
# import schwabttk.price_features
# importlib.reload(schwabttk.price_features)

SYM     = "ZW"
PERIODS = [21, 64, 128]

df  = load_stored(SYM)
out = compute_all_features(df, periods=PERIODS)

# Check features
out[["datetime","Pattern_1","Pattern_3","Regime"]].tail(10)
out[["datetime","Pattern_1","Pattern_2","Pattern_3"]].tail(10)
out[["datetime","DC_21","DC_64","DC_128"]].tail(1).T
out[["datetime","AC1_21","AC5_21","AC11_21"]].tail(1).T
out[["datetime","CorrRR_21","CorrRR_64","CorrRR_128"]].tail(1).T

#%% FEATURE ANALYSIS — full pipeline (steps 1-7)
# import importlib, schwabttk.feature_analysis as fa; importlib.reload(fa)
from schwabttk.feature_analysis import (
    run_full_analysis, feature_report, select_features,
    condition_number, vif_scores,
)

analysis = run_full_analysis(
    out,
    periods=PERIODS,
    pairwise_mi_max_features=64,   # ≥ #candidates → every feature in iteration 1
    pmi_quantile=0.50,
    var_floor_fraction=0.10,
    max_condition_number=30.0,
    min_pc_loading=0.10,
)
report = feature_report(analysis)   # full summary table
sel    = analysis.selection_detail  # full select_features() output

#%% SELECTION 1 — starting point: how collinear is the full candidate set?
# All inputs select_features() needs are cached on `analysis`,
# so later cells can re-run the selection without redoing steps 1-6.
X_sel = analysis.X_std.dropna()

print(f"Rows used           : {len(X_sel)}")
print(f"Candidates          : {X_sel.shape[1]}")
print(f"κ all candidates    : {condition_number(X_sel):.2f}")
print(f"κ after var floor   : {sel['kappa_baseline']:.2f}")
print(f"κ final             : {sel['kappa_final']:.2f}")
print(f"Variance floor      : {sel['var_floor']:.4f}")
print(f"Iterations          : {sel['n_iterations']}")

# VIF of the full set — the features that are most inflated going in
vif_all = vif_scores(X_sel)
vif_all.head(15)

#%% SELECTION 2 — elimination path (audit trail, one row per drop)
path = pd.DataFrame(sel["kappa_path"])
if path.empty:
    print("No MI+κ elimination — κ was already below the target.")
else:
    path = path.set_index("iteration")
    print(path[["feature", "cluster", "κ_before", "κ_after",
                "κ_improvement", "pc_cost", "drop_score",
                "pmi_threshold", "n_features_remaining"]].to_string())

#%% SELECTION 3 — κ path plot: how fast does conditioning improve?
if not path.empty:
    kappa_target = 30.0
    x  = [0] + list(path.index)
    y  = [path["κ_before"].iloc[0]] + list(path["κ_after"])
    lb = ["start"] + list(path["feature"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(width=2), marker=dict(size=8),
        customdata=lb,
        hovertemplate="iter %{x}<br>dropped: %{customdata}"
                      "<br>κ = %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=kappa_target, line_dash="dot", line_width=1)
    lo = min(min(y), kappa_target) * 0.8
    hi = max(max(y), kappa_target) * 1.25
    fig.update_layout(
        title=f"{SYM} — condition number after each drop "
              f"(dotted line: target κ = {kappa_target:g})",
        xaxis_title="Iteration", yaxis_title="κ (log scale)",
        yaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)]),
        template="plotly_white", hovermode="closest",
    )
    show(fig, f"{SYM}_kappa_path")

#%% SELECTION 4 — kept vs dropped, and why
cols = ["mean_pmi", "global_var", "dominant_pc", "pc_loading",
        "cluster", "vif", "dropped_iter", "selected"]

print("── Selected ──")
print(report.loc[report["selected"], cols].sort_values("vif").to_string())

print("\n── Dropped by variance floor ──")
flat = [f for f in sel["dropped_features"]
        if f not in {s["feature"] for s in sel["kappa_path"]}]
print(flat if flat else "none")

print("\n── Dropped by MI+κ (in order) ──")
print(report.loc[report["dropped_iter"].notna(), cols]
            .sort_values("dropped_iter").to_string())

#%% SELECTION 5 — who "represents" each dropped feature?
# For every dropped feature, the surviving feature it shares the most MI with
# (from the full pairwise MI matrix). Checks that nothing unique was thrown away:
# a dropped feature with low max-MI to all survivors lost information.
pmi_full = analysis.pairwise_mi
kept     = analysis.selected_features

rows = []
for f in sel["dropped_features"]:
    if f not in pmi_full.index:
        continue
    mi_to_kept = pmi_full.loc[f, [k for k in kept if k in pmi_full.columns]]
    best = mi_to_kept.idxmax()
    rows.append({
        "dropped"       : f,
        "represented_by": best,
        "MI"            : round(float(mi_to_kept.max()), 4),
        "corr"          : round(float(analysis.corr_matrix.loc[f, best]), 3),
    })
represented = pd.DataFrame(rows).sort_values("MI")
print(represented.to_string(index=False))
# Low MI at the top of this table = features whose information is NOT
# carried by the selected set. Candidates to look at by hand.

#%% SELECTION 6 — redundancy before vs after (|corr| heatmaps)
def _abs_corr_heatmap(C, title):
    return go.Heatmap(
        z=C.abs().values, x=list(C.columns), y=list(C.index),
        zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="|corr|"),
        hovertemplate="%{y} × %{x}<br>|corr| = %{z:.2f}<extra></extra>",
        name=title,
    )

C_all = analysis.corr_matrix
C_sel = analysis.selected_corr

fig = make_subplots(
    rows=1, cols=2, column_widths=[0.62, 0.38],
    subplot_titles=(f"All candidates ({len(C_all)})",
                    f"Selected ({len(C_sel)})"),
)
fig.add_trace(_abs_corr_heatmap(C_all, "all"), row=1, col=1)
h = _abs_corr_heatmap(C_sel, "selected"); h.showscale = False
fig.add_trace(h, row=1, col=2)
fig.update_yaxes(autorange="reversed")
fig.update_layout(title=f"{SYM} — |correlation| before vs after selection",
                  template="plotly_white", height=750)
show(fig, f"{SYM}_corr_before_after")

print(f"Mean |corr| off-diagonal — all: "
      f"{(C_all.abs().values.sum() - len(C_all)) / (len(C_all)**2 - len(C_all)):.3f} | "
      f"selected: "
      f"{(C_sel.abs().values.sum() - len(C_sel)) / max(len(C_sel)**2 - len(C_sel), 1):.3f}")

#%% SELECTION 7 — sensitivity: re-run select_features with other settings
# Reuses the cached inputs (X_std, PCA, variance, pairwise MI).
# NOTE: each MI+κ iteration recomputes pairwise MI on the survivors — with
# ~59 features this is the slow part. Start with a short grid.
KAPPA_GRID = [60.0, 30.0, 15.0]
PMIQ_GRID  = [0.50]            # e.g. [0.25, 0.50, 0.75]

runs = {}
for k in KAPPA_GRID:
    for q in PMIQ_GRID:
        r = select_features(
            X_sel, analysis.pca, analysis.variance_rank, analysis.pairwise_mi,
            pmi_quantile=q,
            max_condition_number=k,
            var_floor_fraction=0.10,
            min_pc_loading=0.10,
        )
        runs[(k, q)] = r

summary = pd.DataFrame([
    {"κ_target": k, "pmi_q": q,
     "n_selected": len(r["selected_features"]),
     "κ_final": round(r["kappa_final"], 2),
     "max_VIF": round(float(r["vif_final"].max()), 2)
                if len(r["vif_final"]) else np.nan,
     "iterations": r["n_iterations"]}
    for (k, q), r in runs.items()
])
print(summary.to_string(index=False))

# Membership matrix: which features survive under which setting
all_feats  = X_sel.columns
membership = pd.DataFrame(
    {f"κ≤{k:g} q={q}": [f in r["selected_features"] for f in all_feats]
     for (k, q), r in runs.items()},
    index=all_feats,
)
membership = membership[membership.any(axis=1)]
membership["n_runs"] = membership.sum(axis=1)
membership.sort_values("n_runs", ascending=False)
# Features kept in every run = the stable core. Features that flip in and out
# are interchangeable members of the same MI cluster.

#%% TREND SCORE :: FEATURE PROCESS
from schwabttk.trend_score import trend_score_pipeline
result    = trend_score_pipeline(out, analysis)
df_scored = result["df_scored"]