# schwabttk/feature_rolling.py
"""
Rolling (walk-forward) Feature Selection
=========================================
Runs the MI+κ feature selection of feature_analysis.select_features() on a
walk-forward schedule, so that the selected set at every refit date uses
ONLY data available at that date. This is the starting point for a
backtest: the trend score at bar t must be built from the set frozen at the
last refit on or before t.

Schedule
--------
    window W  : estimation lookback in bars   (default 1260 ≈ 5 years)
    step   H  : bars between refits           (default 64)
    expanding : anchored window instead of rolling (default False)

    refit k uses rows (t_k − W, t_k] of the standardized feature matrix;
    the resulting set is valid from the close of t_k until t_{k+1}.

Features and their 252-bar rolling z-scores are causal, so they are
computed ONCE on the full history and sliced per window — no lookahead.

What is tracked per refit date × feature
----------------------------------------
Selection outcome (path-dependent — what the backtest uses)
    selected     : bool
    survival     : depth in the elimination — 0 = variance-floor drop,
                   i/(D+1) = dropped at MI+κ iteration i of D, 1 = kept
Path-independent diagnostics on the FULL candidate set (explain flips)
    loo_dlogk    : log κ(all) − log κ(all \\ f)   — conditioning contribution
    log_vif      : log VIF                       — linear redundancy
    max_ic       : max_j sqrt(1 − exp(−2·MI(f,j))) — nonlinear redundancy, [0,1]
    communality  : Σ_k λ_k v_fk² / var(f)         — PC representation, [0,1]

Per refit date
    windows      : n_obs, κ_all, κ_baseline, κ_final, n_selected, ...

Workflow
--------
    res  = rolling_selection(df_features, window=1260, step=64)
    stab = selection_stability(res)
    eff  = effective_selection(res, rule="hysteresis", k_out=2)
    feats_t = selected_as_of(res, t, rule="hysteresis")      # backtest hook
    mask    = expand_to_bars(res, eff)                        # bar × feature
"""

import io
import pickle
import contextlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from schwabttk.feature_analysis import (
    build_feature_matrix, standardize_features, variance_analysis,
    run_pca, pairwise_mi_matrix, select_features, communality,
    _kappa_from_corr,
)

DIAGNOSTICS = ["loo_dlogk", "log_vif", "max_ic", "communality"]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RollingSelectionResult:
    feature_cols : list = field(default_factory=list)
    refit_dates  : Optional[pd.DatetimeIndex] = None
    selected     : Optional[pd.DataFrame] = None   # refit_date × feature, bool
    survival     : Optional[pd.DataFrame] = None   # refit_date × feature, [0,1]
    loo_dlogk    : Optional[pd.DataFrame] = None
    log_vif      : Optional[pd.DataFrame] = None
    max_ic       : Optional[pd.DataFrame] = None
    communality  : Optional[pd.DataFrame] = None
    windows      : Optional[pd.DataFrame] = None   # per-refit metadata
    details      : dict = field(default_factory=dict)  # refit_date → kappa_path etc.
    X_std        : Optional[pd.DataFrame] = None   # full standardized matrix
    params       : dict = field(default_factory=dict)

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"[INFO] Saved rolling selection → {path}")

    @staticmethod
    def load(path: str) -> "RollingSelectionResult":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-WINDOW WORK
# ═══════════════════════════════════════════════════════════════════════════════
def window_diagnostics(Xw: pd.DataFrame,
                       pca_out: dict,
                       pmi: pd.DataFrame) -> pd.DataFrame:
    """
    Path-independent diagnostics on the full candidate set of one window.

    loo_dlogk ≥ 0 always (eigenvalue interlacing: removing a column can only
    shrink λ_max and raise λ_min of the correlation matrix).
    """
    cols = list(Xw.columns)
    C    = np.corrcoef(Xw.values, rowvar=False)
    p    = len(cols)

    log_k_all = np.log(_kappa_from_corr(C))
    dlogk = np.empty(p)
    for i in range(p):
        idx = [j for j in range(p) if j != i]
        dlogk[i] = log_k_all - np.log(_kappa_from_corr(C[np.ix_(idx, idx)]))

    vif = np.diag(np.linalg.pinv(C))
    log_vif = np.log(np.clip(vif, 1.0, None))

    mi = pmi.reindex(index=cols, columns=cols).values.astype(float)
    ic = np.sqrt(1.0 - np.exp(-2.0 * np.clip(mi, 0, None)))
    np.fill_diagonal(ic, np.nan)
    max_ic = np.nanmax(ic, axis=1)

    comm = communality(pca_out, Xw).reindex(cols).values

    return pd.DataFrame({
        "loo_dlogk"  : dlogk,
        "log_vif"    : log_vif,
        "max_ic"     : max_ic,
        "communality": comm,
    }, index=cols)


def _survival_depth(sel: dict, cols: list) -> pd.Series:
    """0 = variance floor, i/(D+1) = MI+κ drop at iteration i of D, 1 = kept."""
    path    = sel["kappa_path"]
    D       = len(path)
    by_iter = {s["feature"]: s["iteration"] for s in path}
    out = {}
    for f in cols:
        if f in sel["selected_features"]:
            out[f] = 1.0
        elif f in by_iter:
            out[f] = by_iter[f] / (D + 1)
        else:
            out[f] = 0.0
    return pd.Series(out)


def _fit_window(Xw: pd.DataFrame, refit_date, p: dict, quiet: bool) -> dict:
    """Full selection + diagnostics for one window (runs in a worker)."""
    sink = io.StringIO() if quiet else None
    ctx  = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        var_out = variance_analysis(Xw, rolling_window=p["variance_window"])
        pca_out = run_pca(Xw, variance_threshold=p["pca_variance_threshold"])
        pmi     = pairwise_mi_matrix(Xw, n_neighbors=p["pairwise_mi_neighbors"],
                                     max_features=Xw.shape[1])
        sel     = select_features(
            Xw, pca_out, var_out, pmi,
            pmi_quantile=p["pmi_quantile"],
            var_floor_fraction=p["var_floor_fraction"],
            max_condition_number=p["max_condition_number"],
            min_pc_loading=p["min_pc_loading"],
            pairwise_mi_neighbors=p["pairwise_mi_neighbors"],
            pc_cost_method=p["pc_cost_method"],
        )
        diag = window_diagnostics(Xw, pca_out, pmi)

    cols = list(Xw.columns)
    C    = np.corrcoef(Xw.values, rowvar=False)
    meta = {
        "window_start": Xw.index[0],
        "n_obs"       : len(Xw),
        "kappa_all"   : _kappa_from_corr(C),
        "kappa_baseline": sel["kappa_baseline"],
        "kappa_final" : sel["kappa_final"],
        "n_selected"  : len(sel["selected_features"]),
        "n_flat"      : len(sel["dropped_features"]) - len(sel["kappa_path"]),
        "n_iterations": sel["n_iterations"],
        "n_pcs"       : pca_out["n_components"],
        "max_vif_sel" : float(sel["vif_final"].max())
                        if len(sel["vif_final"]) else np.nan,
    }
    return {
        "refit_date": refit_date,
        "selected"  : pd.Series({f: f in sel["selected_features"] for f in cols}),
        "survival"  : _survival_depth(sel, cols),
        "diag"      : diag,
        "meta"      : meta,
        "detail"    : {
            "selected_features": sel["selected_features"],
            "dropped_features" : sel["dropped_features"],
            "kappa_path"       : sel["kappa_path"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════════════
def refit_schedule(n_rows: int, window: int, step: int,
                   include_last: bool = False) -> list:
    """Row positions (0-based, in the standardized matrix) of refit bars."""
    if n_rows < window:
        raise ValueError(f"Only {n_rows} usable rows < window={window}. "
                         f"Reduce the window or use a longer history.")
    ends = list(range(window - 1, n_rows, step))
    if include_last and ends[-1] != n_rows - 1:
        ends.append(n_rows - 1)
    return ends


def rolling_selection(df_features: pd.DataFrame,
                      periods: list = None,
                      window: int = 1260,
                      step: int = 64,
                      expanding: bool = False,
                      include_last: bool = False,
                      std_window: int = 252,
                      min_obs: int = 128,
                      date_col: str = "datetime",
                      variance_window: int = 252,
                      pca_variance_threshold: float = 0.90,
                      pairwise_mi_neighbors: int = 5,
                      pmi_quantile: float = 0.50,
                      var_floor_fraction: float = 0.10,
                      max_condition_number: float = 30.0,
                      min_pc_loading: float = 0.10,
                      pc_cost_method: str = "communality",
                      n_jobs: int = -1,
                      verbose: int = 5) -> RollingSelectionResult:
    """
    Walk-forward MI+κ feature selection.

    Parameters
    ----------
    df_features   : output of compute_all_features()
    window        : lookback in bars (1260 ≈ 5y of daily bars)
    step          : bars between refits (64)
    expanding     : True → anchored window starting at the first usable bar
    include_last  : also refit on the final bar (for a live/current set);
                    leaves an irregular last step, so keep False for backtests
    pc_cost_method: "communality" (default here — comparable across windows)
                    or "max_loading" (matches the full-sample default)
    n_jobs        : joblib workers; windows are independent (−1 = all cores)
    verbose       : joblib progress verbosity (0 = silent)

    Other parameters are passed to the per-window pipeline and match
    run_full_analysis().

    Returns
    -------
    RollingSelectionResult
    """
    if periods is None:
        periods = [21, 64, 128]

    # ── Causal features, computed once ────────────────────────────────────
    with contextlib.redirect_stdout(io.StringIO()):
        X     = build_feature_matrix(df_features, periods, min_obs=min_obs)
        X_std = standardize_features(X, window=std_window).dropna()

    if date_col in df_features.columns:
        X_std.index = pd.DatetimeIndex(df_features.loc[X_std.index, date_col])
    else:
        X_std.index = pd.DatetimeIndex(X_std.index)
    X_std = X_std[~X_std.index.duplicated(keep="last")]

    ends  = refit_schedule(len(X_std), window, step, include_last)
    cols  = list(X_std.columns)
    p = dict(variance_window=variance_window,
             pca_variance_threshold=pca_variance_threshold,
             pairwise_mi_neighbors=pairwise_mi_neighbors,
             pmi_quantile=pmi_quantile,
             var_floor_fraction=var_floor_fraction,
             max_condition_number=max_condition_number,
             min_pc_loading=min_pc_loading,
             pc_cost_method=pc_cost_method)

    print(f"[INFO] Rolling selection | {len(cols)} features | "
          f"{len(X_std)} usable bars "
          f"({X_std.index[0].date()} → {X_std.index[-1].date()})")
    print(f"[INFO] window={window} ({'expanding' if expanding else 'rolling'}) "
          f"| step={step} | {len(ends)} refits "
          f"({X_std.index[ends[0]].date()} → {X_std.index[ends[-1]].date()})")

    def _slice(e):
        s = 0 if expanding else e - window + 1
        return X_std.iloc[s:e + 1]

    quiet = True
    if n_jobs == 1:
        outs = []
        for k, e in enumerate(ends):
            outs.append(_fit_window(_slice(e), X_std.index[e], p, quiet))
            if verbose:
                print(f"  refit {k+1}/{len(ends)}  {X_std.index[e].date()}  "
                      f"selected={outs[-1]['meta']['n_selected']}")
    else:
        outs = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_fit_window)(_slice(e), X_std.index[e], p, quiet)
            for e in ends
        )

    dates = pd.DatetimeIndex([o["refit_date"] for o in outs], name="refit_date")

    def _stack(key, sub=None):
        rows = [(o[key][sub] if sub else o[key]) for o in outs]
        return pd.DataFrame(rows, index=dates)[cols]

    res = RollingSelectionResult(
        feature_cols = cols,
        refit_dates  = dates,
        selected     = _stack("selected").astype(bool),
        survival     = _stack("survival"),
        windows      = pd.DataFrame([o["meta"] for o in outs], index=dates),
        details      = {o["refit_date"]: o["detail"] for o in outs},
        X_std        = X_std,
        params       = dict(p, periods=periods, window=window, step=step,
                            expanding=expanding, std_window=std_window,
                            include_last=include_last),
    )
    for d in DIAGNOSTICS:
        setattr(res, d, pd.DataFrame(
            [o["diag"][d] for o in outs], index=dates)[cols])

    w = res.windows
    print(f"[INFO] Done | selected per refit: "
          f"min={w.n_selected.min()} median={w.n_selected.median():.0f} "
          f"max={w.n_selected.max()} | κ_all median={w.kappa_all.median():.1f}")
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def nogueira_stability(S: pd.DataFrame) -> float:
    """
    Nogueira, Sechidis & Brown (2018, JMLR) stability index for a set of
    M selections (rows) over p features (bool columns).
    1 = identical sets; ~0 = no better than random sets of the same sizes.
    """
    Z = S.values.astype(float)
    M, p = Z.shape
    if M < 2:
        return np.nan
    p_hat = Z.mean(axis=0)
    s2    = M / (M - 1) * p_hat * (1 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / p) * (1 - k_bar / p)
    if denom <= 0:
        return np.nan
    return float(1 - s2.mean() / denom)


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def selection_stability(res: RollingSelectionResult,
                        lookback: int = 4,
                        effective: Optional[pd.DataFrame] = None) -> dict:
    """
    Stability of the selected sets through time.

    Parameters
    ----------
    lookback  : number of consecutive refits for rolling Nogueira / frequency
                (4 × 64 bars ≈ 1 year)
    effective : optional output of effective_selection() — when given, the
                timeline also reports its size, turnover and κ.

    Returns
    -------
    dict:
        timeline         : per refit — n_selected, added, removed,
                           jaccard_prev, nogueira_rolling (+ effective cols)
        frequency        : rolling selection frequency (refit × feature)
        overall_frequency: selection frequency over all refits (sorted)
        nogueira_overall : float
    """
    S    = res.selected
    rows = []
    prev = None
    for i, dt in enumerate(S.index):
        cur = set(S.columns[S.loc[dt]])
        blk = S.iloc[max(0, i - lookback + 1): i + 1]
        rows.append({
            "n_selected"      : len(cur),
            "added"           : len(cur - prev) if prev is not None else np.nan,
            "removed"         : len(prev - cur) if prev is not None else np.nan,
            "jaccard_prev"    : _jaccard(cur, prev) if prev is not None else np.nan,
            "nogueira_rolling": nogueira_stability(blk),
        })
        prev = cur
    tl = pd.DataFrame(rows, index=S.index)

    if effective is not None:
        eprev, e_rows = None, []
        for dt in effective.index:
            cur = set(effective.columns[effective.loc[dt]])
            e_rows.append({
                "n_effective"      : len(cur),
                "eff_turnover"     : (len(cur ^ eprev) if eprev is not None
                                      else np.nan),
                "kappa_effective"  : effective_kappa(res, sorted(cur), dt),
            })
            eprev = cur
        tl = tl.join(pd.DataFrame(e_rows, index=effective.index))

    return {
        "timeline"         : tl,
        "frequency"        : S.astype(float).rolling(lookback, min_periods=1).mean(),
        "overall_frequency": S.mean().sort_values(ascending=False),
        "nogueira_overall" : nogueira_stability(S),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST HOOKS
# ═══════════════════════════════════════════════════════════════════════════════
def effective_selection(res: RollingSelectionResult,
                        rule: str = "hysteresis",
                        k_out: int = 2,
                        min_freq: float = 0.5,
                        lookback: int = 4,
                        respect_kappa: bool = True) -> pd.DataFrame:
    """
    Turn raw per-refit selections into the set actually used downstream.

    rule
    ----
    "raw"        : the set selected at each refit, unchanged
    "hysteresis" : a feature enters on first selection and leaves only after
                   being unselected for k_out consecutive refits
    "frequency"  : features selected in ≥ min_freq of the last `lookback`
                   refits (uses only past and current refits)

    respect_kappa (hysteresis only)
        Carrying unselected features can push κ of the effective set back
        above max_condition_number. When True, carried-over features (never
        the freshly selected ones) are removed greedily — largest κ
        reduction first — until κ ≤ target on the current window.

    Returns
    -------
    pd.DataFrame — refit_date × feature, bool
    """
    S = res.selected
    if rule == "raw":
        return S.copy()

    if rule == "frequency":
        f = S.astype(float).rolling(lookback, min_periods=1).mean()
        return f >= min_freq

    if rule == "hysteresis":
        state, misses, rows = set(), {}, []
        for dt in S.index:
            cur = set(S.columns[S.loc[dt]])
            for f in cur:
                state.add(f)
                misses[f] = 0
            for f in list(state - cur):
                misses[f] = misses.get(f, 0) + 1
                if misses[f] >= k_out:
                    state.discard(f)
            if respect_kappa:
                target  = res.params["max_condition_number"]
                carried = state - cur
                while carried and \
                        effective_kappa(res, sorted(state), dt) > target:
                    best = min(carried, key=lambda f: effective_kappa(
                        res, sorted(state - {f}), dt))
                    state.discard(best)
                    carried.discard(best)
            rows.append({c: c in state for c in S.columns})
        return pd.DataFrame(rows, index=S.index)

    raise ValueError(f"unknown rule '{rule}'")


def effective_kappa(res: RollingSelectionResult, feats: list, refit_date) -> float:
    """κ of an arbitrary feature set on the window that ends at refit_date."""
    if len(feats) <= 1:
        return 1.0
    e = res.X_std.index.get_loc(refit_date)
    s = 0 if res.params["expanding"] else e - res.params["window"] + 1
    Xw = res.X_std.iloc[s:e + 1][feats]
    return _kappa_from_corr(np.corrcoef(Xw.values, rowvar=False))


def selected_as_of(res: RollingSelectionResult,
                   t,
                   rule: str = "hysteresis",
                   **rule_kw) -> list:
    """
    Feature set to use at bar t: the effective set frozen at the last refit
    on or before t. The refit at date d uses data through the close of d,
    so its set is valid for decisions made at/after that close.

    Raises ValueError if t precedes the first refit (warm-up period).
    """
    eff  = effective_selection(res, rule=rule, **rule_kw)
    t    = pd.Timestamp(t)
    past = eff.index[eff.index <= t]
    if len(past) == 0:
        raise ValueError(f"{t.date()} is before the first refit "
                         f"({eff.index[0].date()}) — no out-of-sample set yet.")
    row = eff.loc[past[-1]]
    return list(row.index[row])


def expand_to_bars(res: RollingSelectionResult,
                   effective: pd.DataFrame,
                   dates: Optional[pd.DatetimeIndex] = None,
                   lag: int = 0) -> pd.DataFrame:
    """
    Forward-fill the per-refit effective sets onto every bar.

    dates : bar dates to expand to (default: all standardized-matrix bars)
    lag   : extra bars to delay each set (0 = usable from the refit close;
            1 = usable from the next bar)

    Returns bar × feature bool DataFrame; bars before the first refit are
    all False.
    """
    if dates is None:
        dates = res.X_std.index
    dates = pd.DatetimeIndex(dates)
    m = effective.reindex(dates.union(effective.index)).ffill()
    m = m.reindex(dates)
    if lag:
        m = m.shift(lag)
    return m.fillna(False).astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS (plotly)
# ═══════════════════════════════════════════════════════════════════════════════
def _feature_order(res, by="frequency"):
    if by == "frequency":
        return list(res.selected.mean().sort_values(ascending=False).index)
    return list(res.feature_cols)


def plot_selection_heatmap(res: RollingSelectionResult,
                           effective: Optional[pd.DataFrame] = None,
                           title: str = None):
    """
    Feature × refit-date heatmap of survival depth (1 = selected).
    Features sorted by overall selection frequency (most stable on top).
    If `effective` is given, cells in the effective set but NOT raw-selected
    (hysteresis carry-overs) are outlined in the hover text.
    """
    import plotly.graph_objects as go

    order = _feature_order(res)
    Z     = res.survival[order].T
    sel   = res.selected[order].T
    freq  = res.selected[order].mean()
    ylab  = [f"{f}  ({freq[f]:.0%})" for f in order]

    txt = np.where(sel.values, "selected", "dropped")
    if effective is not None:
        eff = effective[order].T.values
        txt = np.where(eff & ~sel.values, "carried (hysteresis)", txt)

    fig = go.Figure(go.Heatmap(
        z=Z.values, x=Z.columns, y=ylab,
        zmin=0, zmax=1, colorscale="Blues",
        colorbar=dict(title="survival<br>depth",
                      tickvals=[0, 0.5, 1],
                      ticktext=["floor", "mid", "kept"]),
        customdata=txt,
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>depth=%{z:.2f}"
                      "<br>%{customdata}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        title=title or "Rolling selection — survival depth "
                       "(label = overall selection frequency)",
        template="plotly_white",
        height=max(420, 16 * len(order) + 140),
    )
    return fig


def plot_diagnostic_heatmap(res: RollingSelectionResult, name: str,
                            title: str = None):
    """Feature × refit-date heatmap of one path-independent diagnostic."""
    import plotly.graph_objects as go
    if name not in DIAGNOSTICS:
        raise ValueError(f"name must be one of {DIAGNOSTICS}")
    labels = {
        "loo_dlogk"  : "Δlog κ if removed",
        "log_vif"    : "log VIF",
        "max_ic"     : "max IC to any other",
        "communality": "communality",
    }
    order = _feature_order(res)
    Z = getattr(res, name)[order].T
    zmax = 1 if name in ("max_ic", "communality") else float(np.nanquantile(Z.values, 0.98))
    fig = go.Figure(go.Heatmap(
        z=Z.values, x=Z.columns, y=order,
        zmin=0, zmax=zmax, colorscale="Blues",
        colorbar=dict(title=labels[name]),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>"
                      + labels[name] + "=%{z:.3f}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title or f"Rolling diagnostic — {labels[name]}",
                      template="plotly_white",
                      height=max(420, 16 * len(order) + 140))
    return fig


def plot_stability(res: RollingSelectionResult, stab: dict, title: str = None):
    """
    Three stacked panels on a shared time axis (no dual y-axes):
      1. κ of all candidates vs κ of the final selection (log) + target
      2. number of features — raw selection (and effective set if present)
      3. Jaccard vs previous refit and rolling Nogueira stability, [0,1]
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    w, tl = res.windows, stab["timeline"]
    target = res.params["max_condition_number"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=("Condition number κ (log)",
                                        "Features in set",
                                        "Set stability"))
    c1, c2 = "#2a6fdb", "#d9822b"

    fig.add_trace(go.Scatter(x=w.index, y=w.kappa_all, name="all candidates",
                             line=dict(color=c1, width=2), legendgroup="k",
                             legendgrouptitle_text="κ"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=w.index, y=w.kappa_final, name="selected",
                             line=dict(color=c2, width=2), legendgroup="k"),
                  row=1, col=1)
    if "kappa_effective" in tl:
        fig.add_trace(go.Scatter(x=tl.index, y=tl.kappa_effective,
                                 name="effective set",
                                 line=dict(color=c2, width=2, dash="dot"),
                                 legendgroup="k"), row=1, col=1)
    fig.add_hline(y=target, line_dash="dot", line_width=1, row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    fig.add_trace(go.Scatter(x=tl.index, y=tl.n_selected, name="raw selection",
                             line=dict(color=c1, width=2, shape="hv"),
                             legendgroup="n", legendgrouptitle_text="Features in set"),
                  row=2, col=1)
    if "n_effective" in tl:
        fig.add_trace(go.Scatter(x=tl.index, y=tl.n_effective,
                                 name="effective set",
                                 line=dict(color=c2, width=2, shape="hv"),
                                 legendgroup="n"),
                      row=2, col=1)

    fig.add_trace(go.Scatter(x=tl.index, y=tl.jaccard_prev,
                             name="Jaccard vs previous",
                             line=dict(color=c1, width=2),
                             legendgroup="s", legendgrouptitle_text="Stability"), row=3, col=1)
    fig.add_trace(go.Scatter(x=tl.index, y=tl.nogueira_rolling,
                             name="Nogueira (rolling)",
                             line=dict(color=c2, width=2), legendgroup="s"), row=3, col=1)
    fig.update_yaxes(range=[0, 1.02], row=3, col=1)

    fig.update_layout(
        title=title or f"Rolling selection stability — overall Nogueira "
                       f"= {stab['nogueira_overall']:.2f}",
        template="plotly_white", height=820, hovermode="x unified",
        legend=dict(groupclick="toggleitem", tracegroupgap=150, y=1, yanchor="top"),
    )
    return fig


def plot_feature_history(res: RollingSelectionResult, feature: str):
    """One feature through time: survival depth + the four diagnostics."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    panels = [("survival", "survival depth (1 = kept)")] + [
        ("loo_dlogk", "Δlog κ if removed"), ("log_vif", "log VIF"),
        ("max_ic", "max IC to any other"), ("communality", "communality")]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                        vertical_spacing=0.035,
                        subplot_titles=[p[1] for p in panels])
    for i, (attr, _) in enumerate(panels, start=1):
        s = getattr(res, attr)[feature]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines+markers",
            line=dict(width=2, color="#2a6fdb",
                      shape="hv" if attr == "survival" else "linear"),
            marker=dict(size=6), showlegend=False,
        ), row=i, col=1)
    fig.update_layout(title=f"{feature} — rolling selection history",
                      template="plotly_white", height=170 * len(panels) + 80,
                      hovermode="x unified")
    return fig