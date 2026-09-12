# schwabttk/price_features.py
"""
Futures Feature Engineering
============================
Deterministic, stateless pipeline: OHLCV → features.
Same input always produces the same output — no fitting required.

Group 1 — Candle Anatomy       : Body, Wicks, Range, Ratios, CLV
Group 2 — Returns              : r_day, r_night, r_total (log returns)
Group 3 — Rolling Features     : R, RV, TM, ER, DC, RVOL, VTC, VWR, VMD,
                                  AC1/AC5/AC21, Run statistics
Group 4 — Price vs EMA         : EMA, PVE, PVE_pct, EMA_slope
Group 5 — Conviction & Momentum: Conviction, MOM
Group 6 — Pattern Classifiers  : single, double, triple, N-candle (Regime)
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Raise informative error if a required column is missing."""
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found. "
            f"Check that the required pipeline step has been run first."
        )
    return df[col]


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — CANDLE ANATOMY
# ═══════════════════════════════════════════════════════════════════════════════
def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar candle anatomy features.

    Features
    --------
    Body       : |Close - Open|
    UpperWick  : High  - max(Close, Open)
    LowerWick  : min(Close, Open) - Low
    Range      : High - Low
    BodyRatio  : Body  / Range          ∈ [0, 1]
    UpperRatio : UpperWick / Range      ∈ [0, 1]
    LowerRatio : LowerWick / Range      ∈ [0, 1]
    CLV        : (2*Close - High - Low) / Range   ∈ [-1, +1]
    """
    out = df.copy()
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]

    out["Body"]      = (c - o).abs()
    out["UpperWick"] = h - np.maximum(c, o)
    out["LowerWick"] = np.minimum(c, o) - l
    out["Range"]     = h - l

    # Guard against zero-range bars (e.g. halted sessions)
    rng = out["Range"].replace(0, np.nan)
    out["BodyRatio"]  = out["Body"]      / rng
    out["UpperRatio"] = out["UpperWick"] / rng
    out["LowerRatio"] = out["LowerWick"] / rng
    out["CLV"]        = (2 * c - h - l)  / rng

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — RETURNS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-bar log return series.

    r_day   = log(Close/Open)           roll-free intraday return
    r_night = log(Open/lag(Close))      overnight (may span roll gaps)
    r_total = log(Close/lag(Close))     standard close-to-close
    """
    out = df.copy()
    out["r_day"]   = np.log(out["close"] / out["open"])
    out["r_night"] = np.log(out["open"]  / out["close"].shift(1))
    out["r_total"] = np.log(out["close"] / out["close"].shift(1))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — ROLLING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def rolling_features(df: pd.DataFrame,
                     periods: list = None) -> pd.DataFrame:
    """
    Compute rolling features for each period p in `periods`.

    Per-period columns added (sfx = _{p})
    --------------------------------------
    R_{p}         : cumulative log r_day over p bars (additive)
    RV_{p}        : realized volatility = sqrt(sum r_day^2)
    TM_{p}        : trend magnitude = R_{p} / RV_{p}  ∈ [-1, +1]
    ER_{p}        : efficiency ratio = |R_{p}| / sum|r_day|
    DC_{p}        : directional consistency — fraction of bars where
                    sign(r_day) matches sign(R_{p}) at window end
    AC1_{p}       : lag-1 autocorrelation of r_day over p bars
    AC5_{p}       : lag-5 autocorrelation of r_day over p bars
    AC21_{p}      : lag-21 autocorrelation of r_day over p bars
    nRunPos_{p}   : number of positive return runs in window
    nRunNeg_{p}   : number of negative return runs in window
    AvgRunPos_{p} : average positive streak length
    AvgRunNeg_{p} : average negative streak length
    MaxRunPos_{p} : longest positive streak
    MaxRunNeg_{p} : longest negative streak
    DomRun_{p}    : dominant run direction = sign(AvgRunPos - AvgRunNeg)
    RVOL_{p}      : relative volume = vol(t) / mean(vol, p)
    VTC_{p}       : corr(r_day, dVol) over p bars
    VWR_{p}       : RVOL-weighted mean r_day over p bars
    VMD_{p}       : volume-momentum divergence flag

    Once (not per period)
    ---------------------
    dVol          : volume change = vol(t) - vol(t-1)
    """
    if periods is None:
        periods = [21, 64, 128]

    out   = df.copy()
    r     = _safe_col(out, "r_day")
    vol   = _safe_col(out, "volume")
    sign_r = np.sign(r)

    # Volume change — computed once
    out["dVol"] = vol.diff()

    # ── Run statistics helper ─────────────────────────────────────────────
    def run_stats(x: np.ndarray) -> np.ndarray:
        """
        Compute run statistics for a window of returns.
        Returns [n_pos, n_neg, avg_pos, avg_neg, max_pos, max_neg, dom_dir]
        """
        s = np.sign(x)
        s = s[s != 0]               # remove flat bars
        if len(s) < 3:
            return np.array([np.nan] * 7)

        changes    = np.concatenate(([1], np.diff(s) != 0, [1]))
        run_starts = np.where(changes[:-1])[0]
        run_ends   = np.where(changes[1:])[0]
        run_lens   = run_ends - run_starts + 1
        run_signs  = s[run_starts]

        pos_lens = run_lens[run_signs > 0]
        neg_lens = run_lens[run_signs < 0]

        n_pos   = len(pos_lens)
        n_neg   = len(neg_lens)
        avg_pos = pos_lens.mean() if n_pos > 0 else 0.0
        avg_neg = neg_lens.mean() if n_neg > 0 else 0.0
        max_pos = pos_lens.max()  if n_pos > 0 else 0.0
        max_neg = neg_lens.max()  if n_neg > 0 else 0.0
        dom_dir = np.sign(avg_pos - avg_neg)

        return np.array([n_pos, n_neg, avg_pos, avg_neg,
                         max_pos, max_neg, dom_dir])

    for p in periods:
        sfx = f"_{p}"

        # ── Cumulative return (log returns add) ───────────────────────────
        out[f"R{sfx}"] = r.rolling(p).sum()

        # ── Realized Volatility ───────────────────────────────────────────
        out[f"RV{sfx}"] = np.sqrt(r.pow(2).rolling(p).sum())

        # ── Trend Magnitude ───────────────────────────────────────────────
        out[f"TM{sfx}"] = (out[f"R{sfx}"]
                           / out[f"RV{sfx}"].replace(0, np.nan))

        # ── Efficiency Ratio ──────────────────────────────────────────────
        sum_abs_r = r.abs().rolling(p).sum().replace(0, np.nan)
        out[f"ER{sfx}"] = out[f"R{sfx}"].abs() / sum_abs_r

        # ── Directional Consistency (vectorized) ──────────────────────────
        # At each t, sign(R_{p}) is the net direction of that window.
        # DC = fraction of bars where sign(r_day) matches sign(R_{p,t}).
        sign_R_t  = np.sign(out[f"R{sfx}"])
        pos_frac  = (sign_r > 0).astype(float).rolling(p).mean()
        neg_frac  = (sign_r < 0).astype(float).rolling(p).mean()
        zero_frac = (sign_r == 0).astype(float).rolling(p).mean()
        out[f"DC{sfx}"] = np.where(
            sign_R_t ==  1, pos_frac,
            np.where(sign_R_t == -1, neg_frac, zero_frac)
        )

        # ── Autocorrelation ───────────────────────────────────────────────
        for q in [1, 5, 21]:
            out[f"AC{q}{sfx}"] = (
                r.rolling(p)
                .apply(
                    lambda x: pd.Series(x).autocorr(lag=q)
                    if len(x) > q + 1 else np.nan,
                    raw=False
                )
            )

        # ── Run Statistics ────────────────────────────────────────────────
        run_cols    = [
            f"nRunPos{sfx}",  f"nRunNeg{sfx}",
            f"AvgRunPos{sfx}", f"AvgRunNeg{sfx}",
            f"MaxRunPos{sfx}", f"MaxRunNeg{sfx}",
            f"DomRun{sfx}"
        ]
        run_results = np.full((len(out), 7), np.nan)
        for i in range(p - 1, len(out)):
            run_results[i] = run_stats(r.iloc[i - p + 1: i + 1].values)
        for j, col in enumerate(run_cols):
            out[col] = run_results[:, j]

        # ── Relative Volume ───────────────────────────────────────────────
        out[f"RVOL{sfx}"] = vol / vol.rolling(p).mean()

        # ── Volume Trend Confirmation: corr(r_day, dVol) ─────────────────
        out[f"VTC{sfx}"] = r.rolling(p).corr(out["dVol"])

        # ── RVOL-Weighted Return ──────────────────────────────────────────
        rvol = out[f"RVOL{sfx}"]
        num  = (r * rvol).rolling(p).sum()
        den  = rvol.rolling(p).sum().replace(0, np.nan)
        out[f"VWR{sfx}"] = num / den

        # ── Volume Momentum Divergence ────────────────────────────────────
        out[f"VMD{sfx}"] = (
            np.sign(out[f"R{sfx}"]) != np.sign(out[f"VTC{sfx}"])
        ).astype(float)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — PRICE VS EMA
# ═══════════════════════════════════════════════════════════════════════════════
def price_vs_ema(df: pd.DataFrame,
                 periods: list = None) -> pd.DataFrame:
    """
    Compute price position relative to EMA for each period p.

    Per-period columns added
    ------------------------
    EMA_{p}       : exponential moving average of close
    PVE_{p}       : Close / EMA_{p} - 1   (return-like deviation)
                    > 0 → price above EMA, < 0 → below, ~0 → at average
    PVE_pct_{p}   : rolling percentile of PVE_{p} over p bars ∈ [0, 100]
                    100 → most extended above EMA in window history
                    0   → most extended below EMA in window history
                    50  → at median (neither extreme)
    EMA_slope_{p} : log(EMA_{p} / EMA_{p}.shift(1)) — EMA own direction
    """
    if periods is None:
        periods = [21, 64, 128]

    out = df.copy()
    c   = out["close"]

    for p in periods:
        sfx = f"_{p}"

        ema              = c.ewm(span=p, adjust=False).mean()
        out[f"EMA{sfx}"] = ema

        pve              = c / ema - 1
        out[f"PVE{sfx}"] = pve

        out[f"PVE_pct{sfx}"] = pve.rolling(p).apply(
            lambda x: percentileofscore(x, x.iloc[-1], kind="rank"),
            raw=False
        )

        out[f"EMA_slope{sfx}"] = np.log(ema / ema.shift(1))

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — CONVICTION & MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════
def _estimate_doji_threshold(body_ratio: pd.Series,
                             percentile: float = 20.0) -> float:
    """
    Auto-estimate the doji threshold from the BodyRatio distribution.
    Uses the p-th percentile of non-NaN values.
    Default p=20 captures the lower tail of indecisive bars.
    """
    return float(np.nanpercentile(body_ratio.dropna().values, percentile))


def conviction(df: pd.DataFrame,
               doji_threshold: float = None,
               doji_percentile: float = 20.0) -> pd.DataFrame:
    """
    Compute per-bar Conviction score.

    Logic
    -----
    Step 1 — Doji filter:
        if BodyRatio < doji_threshold → is_doji = True

    Step 2 — Direction (all bars):
        dir = sign(Close - Open)

    Step 3 — Confirmation signals (each ∈ {0, 1}):
        clv_confirm  = 1 if sign(CLV) == dir
        wick_confirm:
          dir > 0 → UpperRatio < BodyRatio  (upper wick not dominating body)
          dir < 0 → LowerRatio < BodyRatio  (lower wick not dominating body)

    Step 4 — Score for non-doji bars ∈ {-1, -0.5, 0, 0.5, +1}:
        score = dir × mean(clv_confirm, wick_confirm)

    Step 5 — Doji refinement ∈ {-0.25, 0, +0.25}:
        Pure doji         → 0.0
        Bull doji: dir>0 AND lwr>br AND lwr>uwr → +0.25
        Bear doji: dir<0 AND uwr>br AND uwr>lwr → -0.25

    Columns added
    -------------
    doji_threshold : scalar threshold used (constant column)
    is_doji        : bool
    is_bull_doji   : bool — doji with bullish wick bias
    is_bear_doji   : bool — doji with bearish wick bias
    Conviction     : float ∈ {-1, -0.5, -0.25, 0, +0.25, +0.5, +1}
    """
    out = df.copy()

    br  = _safe_col(out, "BodyRatio")
    clv = _safe_col(out, "CLV")
    uwr = _safe_col(out, "UpperRatio")
    lwr = _safe_col(out, "LowerRatio")
    o   = out["open"]
    c   = out["close"]

    if doji_threshold is None:
        doji_threshold = _estimate_doji_threshold(br, doji_percentile)
    out["doji_threshold"] = doji_threshold

    # Step 1 — Doji flag
    out["is_doji"] = br < doji_threshold

    # Step 2 — Direction
    direction = np.sign(c - o)

    # Step 3 — Confirmation signals
    clv_confirm  = (np.sign(clv) == direction).astype(float)
    wick_confirm = np.where(
        direction > 0, (uwr < br).astype(float),
        np.where(direction < 0, (lwr < br).astype(float), 0.0)
    )
    wick_confirm = pd.Series(wick_confirm, index=out.index)

    # Step 4 — Raw score for non-doji bars
    confirm_mean = (clv_confirm + wick_confirm) / 2.0
    raw_score    = direction * confirm_mean

    # Step 5 — Doji refinement
    bull_doji = (
        out["is_doji"] & (direction > 0) & (lwr > br) & (lwr > uwr)
    )
    bear_doji = (
        out["is_doji"] & (direction < 0) & (uwr > br) & (uwr > lwr)
    )
    out["is_bull_doji"] = bull_doji
    out["is_bear_doji"] = bear_doji

    conv = raw_score.where(~out["is_doji"], 0.0)  # zero all doji
    conv = conv.where(~bull_doji,  0.25)           # re-assign bull doji
    conv = conv.where(~bear_doji, -0.25)           # re-assign bear doji
    out["Conviction"] = conv

    print(f"[INFO] Doji threshold (BodyRatio): {doji_threshold:.4f} "
          f"({out['is_doji'].sum()} bars = "
          f"{out['is_doji'].mean()*100:.1f}%)")

    return out


def momentum(df: pd.DataFrame,
             periods: list = None) -> pd.DataFrame:
    """
    Compute conviction-scaled momentum for each period p.

    MOM_{p} = R_{p} × mean(Conviction, p)

    Doji bars contribute 0 to the rolling mean so only
    genuinely directional runs produce large MOM values.
    """
    if periods is None:
        periods = [21, 64, 128]

    out  = df.copy()
    conv = _safe_col(out, "Conviction")

    for p in periods:
        sfx = f"_{p}"
        if f"R{sfx}" not in out.columns:
            raise ValueError(
                f"R{sfx} not found. Run rolling_features() first."
            )
        out[f"MOM{sfx}"] = out[f"R{sfx}"] * conv.rolling(p).mean()

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — PATTERN CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════
def classify_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each bar into a single-candle pattern.

    Conviction → Pattern mapping
    ----------------------------
    ±1.0  Strong directional (both CLV and wick confirm)
    ±0.5  Moderate directional (one confirmation)
    ±0.25 Directional doji (doji body, biased wick structure)
     0.0  Pure indecision / neutral doji

    Patterns (evaluated in priority order)
    ----------------------------------------
    StrongHammer       : BullDoji  + LowerRatio > 0.5
    StrongShootingStar : BearDoji  + UpperRatio > 0.5
    BullDoji           : is_bull_doji
    BearDoji           : is_bear_doji
    Hammer             : Conviction ≥ 1.0 + LowerRatio > 0.5
    ShootingStar       : Conviction ≤ -1.0 + UpperRatio > 0.5
    Bull               : Conviction ≥ 1.0
    Bear               : Conviction ≤ -1.0
    Hammer             : Conviction ≥ 0.5 + LowerRatio > 0.5
    ShootingStar       : Conviction ≤ -0.5 + UpperRatio > 0.5
    WeakBull           : Conviction ≥ 0.5
    WeakBear           : Conviction ≤ -0.5
    Indecision         : otherwise
    """
    out  = df.copy()
    conv = _safe_col(out, "Conviction")
    ur   = _safe_col(out, "UpperRatio")
    lr   = _safe_col(out, "LowerRatio")

    is_bull_doji = out.get("is_bull_doji",
                           pd.Series(False, index=out.index))
    is_bear_doji = out.get("is_bear_doji",
                           pd.Series(False, index=out.index))

    conditions = [
        is_bull_doji & (lr > 0.5),
        is_bear_doji & (ur > 0.5),
        is_bull_doji,
        is_bear_doji,
        (conv >= 1.0) & (lr > 0.5),
        (conv <= -1.0) & (ur > 0.5),
        conv >= 1.0,
        conv <= -1.0,
        (conv >= 0.5) & (lr > 0.5),
        (conv <= -0.5) & (ur > 0.5),
        conv >= 0.5,
        conv <= -0.5,
    ]
    choices = [
        "StrongHammer", "StrongShootingStar",
        "BullDoji",     "BearDoji",
        "Hammer",       "ShootingStar",
        "Bull",         "Bear",
        "Hammer",       "ShootingStar",
        "WeakBull",     "WeakBear",
    ]

    out["Pattern_1"] = np.select(conditions, choices, default="Indecision")
    return out


def classify_double(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify consecutive bar pairs into two-candle patterns.

    Patterns (priority order)
    -------------------------
    BearControlShift  : prev bull-like → BearDoji or seller-entry wick
    BullControlShift  : prev bear-like → BullDoji or buyer-entry wick
    Engulfing_Bull    : prev Bear/BearDoji, curr Bull, expanding range
    Engulfing_Bear    : prev Bull/BullDoji, curr Bear, expanding range
    Harami_Bull       : prev Bear, curr Bull, contracting range
    Harami_Bear       : prev Bull, curr Bear, contracting range
    Buildup_Bull      : BullDoji → Bull
    Buildup_Bear      : BearDoji → Bear
    Fading_Bull       : Bull → BullDoji
    Fading_Bear       : Bear → BearDoji
    Continuation_Bull : prev Bull, curr Bull
    Continuation_Bear : prev Bear, curr Bear
    Indecision        : otherwise
    """
    out    = df.copy()
    conv   = _safe_col(out, "Conviction")
    rng    = _safe_col(out, "Range")
    uwr    = _safe_col(out, "UpperRatio")
    lwr    = _safe_col(out, "LowerRatio")
    br     = _safe_col(out, "BodyRatio")
    p_conv = conv.shift(1)
    p_rng  = rng.shift(1)

    is_bull_doji   = out.get("is_bull_doji",
                             pd.Series(False, index=out.index))
    is_bear_doji   = out.get("is_bear_doji",
                             pd.Series(False, index=out.index))
    p_is_bull_doji = is_bull_doji.shift(1)
    p_is_bear_doji = is_bear_doji.shift(1)

    is_bull     = conv   >= 0.5
    is_bear     = conv   <= -0.5
    p_is_bull   = p_conv >= 0.5
    p_is_bear   = p_conv <= -0.5
    expanding   = rng > p_rng
    contracting = rng < p_rng
    p_any_bull  = p_conv > 0
    p_any_bear  = p_conv < 0

    # Control shift: wick dominating body, direction not strongly opposing
    seller_entry = (uwr > br) & (conv <= 0.25)
    buyer_entry  = (lwr > br) & (conv >= -0.25)

    conditions = [
        p_any_bull & (is_bear_doji | seller_entry),
        p_any_bear & (is_bull_doji | buyer_entry),
        (p_is_bear | p_is_bear_doji) & is_bull & expanding,
        (p_is_bull | p_is_bull_doji) & is_bear & expanding,
        p_is_bear & is_bull & contracting,
        p_is_bull & is_bear & contracting,
        p_is_bull_doji & is_bull,
        p_is_bear_doji & is_bear,
        p_is_bull & is_bull_doji,
        p_is_bear & is_bear_doji,
        p_is_bull & is_bull,
        p_is_bear & is_bear,
    ]
    choices = [
        "BearControlShift", "BullControlShift",
        "Engulfing_Bull",   "Engulfing_Bear",
        "Harami_Bull",      "Harami_Bear",
        "Buildup_Bull",     "Buildup_Bear",
        "Fading_Bull",      "Fading_Bear",
        "Continuation_Bull","Continuation_Bear",
    ]

    out["Pattern_2"] = np.select(conditions, choices, default="Indecision")
    return out


def classify_triple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify consecutive bar triplets into three-candle patterns.

    Patterns (priority order)
    -------------------------
    BearCS_Confirmed   : BullLike  → BearControlShift → Bear
    BullCS_Confirmed   : BearLike  → BullControlShift → Bull
    MorningStar_Strong : Bear → PureDoji  → Bull
    MorningStar        : Bear → BearDoji  → Bull
    MorningStar_Weak   : Bear → Indecision → Bull
    EveningStar_Strong : Bull → PureDoji  → Bear
    EveningStar        : Bull → BullDoji  → Bear
    EveningStar_Weak   : Bull → Indecision → Bear
    ThreeWhite         : three consecutive Bull + rising closes
    ThreeCrows         : three consecutive Bear + falling closes
    BullAccel          : WeakBull → Bull → Bull
    BearAccel          : WeakBear → Bear → Bear
    Indecision         : otherwise
    """
    out  = df.copy()
    conv = _safe_col(out, "Conviction")
    c    = out["close"]
    p2   = out.get("Pattern_2",
                   pd.Series("Indecision", index=out.index))

    is_bull_doji = out.get("is_bull_doji",
                           pd.Series(False, index=out.index))
    is_bear_doji = out.get("is_bear_doji",
                           pd.Series(False, index=out.index))

    is_bull      = conv >= 0.5
    is_weak_bull = conv == 0.5
    is_bear      = conv <= -0.5
    is_weak_bear = conv == -0.5
    is_pure_doji = conv == 0.0
    is_indec     = conv.abs() < 0.5

    bull_2      = is_bull.shift(2)
    bull_1      = is_bull.shift(1)
    weak_bull_2 = is_weak_bull.shift(2)
    bear_2      = is_bear.shift(2)
    bear_1      = is_bear.shift(1)
    weak_bear_2 = is_weak_bear.shift(2)
    pure_doji_1 = is_pure_doji.shift(1)
    bear_doji_1 = is_bear_doji.shift(1)
    bull_doji_1 = is_bull_doji.shift(1)
    indec_1     = is_indec.shift(1)
    bear_cs_1   = (p2.shift(1) == "BearControlShift")
    bull_cs_1   = (p2.shift(1) == "BullControlShift")
    any_bull_2  = conv.shift(2) > 0
    any_bear_2  = conv.shift(2) < 0

    rising_closes  = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    falling_closes = (c < c.shift(1)) & (c.shift(1) < c.shift(2))

    conditions = [
        any_bull_2 & bear_cs_1 & is_bear,
        any_bear_2 & bull_cs_1 & is_bull,
        bear_2 & pure_doji_1 & is_bull,
        bear_2 & bear_doji_1 & is_bull,
        bear_2 & indec_1     & is_bull,
        bull_2 & pure_doji_1 & is_bear,
        bull_2 & bull_doji_1 & is_bear,
        bull_2 & indec_1     & is_bear,
        bull_2 & bull_1 & is_bull & rising_closes,
        bear_2 & bear_1 & is_bear & falling_closes,
        weak_bull_2 & bull_1 & is_bull,
        weak_bear_2 & bear_1 & is_bear,
    ]
    choices = [
        "BearCS_Confirmed",   "BullCS_Confirmed",
        "MorningStar_Strong", "MorningStar",     "MorningStar_Weak",
        "EveningStar_Strong", "EveningStar",     "EveningStar_Weak",
        "ThreeWhite",         "ThreeCrows",
        "BullAccel",          "BearAccel",
    ]

    out["Pattern_3"] = np.select(conditions, choices, default="Indecision")
    return out


def classify_n(df: pd.DataFrame,
               periods: list = None,
               tm_flat_thresh: float = 0.10,
               rv_spike_pct: float = 1.5,
               ac1_reverting_thresh: float = -0.10,
               pve_pct_upper: float = 90.0,
               pve_pct_lower: float = 10.0,
               pve_ema_band: float = 0.01,
               ema_slope_flat: float = 0.001) -> pd.DataFrame:
    """
    Classify the N-bar regime using TM cross-period signals,
    run structure, autocorrelation, and price vs EMA position.

    Requires exactly 3 periods [short, mid, long].

    Primary signal  : TM_{p} signs across all three periods
    Secondary       : AC1, DomRun, RV, VMD, PVE_pct, EMA_slope

    Parameters
    ----------
    tm_flat_thresh       : |TM| below this → flat (default 0.10)
    rv_spike_pct         : RV / rolling_mean(RV) for capitulation (1.5)
    ac1_reverting_thresh : AC1 below this → mean-reverting (-0.10)
    pve_pct_upper        : PVE_pct above this → overextended up (90)
    pve_pct_lower        : PVE_pct below this → overextended down (10)
    pve_ema_band         : tolerance band around LT EMA (0.01 = 1%)
    ema_slope_flat       : |EMA_slope| below this → EMA flat (0.001)

    Regimes (priority order)
    ------------------------
    TrendExhaustion   : trend aligned + ST run opposing + overextended
    TrendCapitulation : TM flip ST+IT + RV spike + counter runs dominant
    Accumulation      : LT bear + ST/IT bull + pos runs + price near LT EMA
    Distribution      : LT bull + ST/IT bear + neg runs + price near LT EMA
    Range             : all TM flat + AC1 mean-reverting + EMA flat
    TrendUp           : TM_s>0, TM_m>0, TM_l>0
    TrendDown         : TM_s<0, TM_m<0, TM_l<0
    Retracement_Bull  : TM_s<0, TM_m>0, TM_l>0
    Retracement_Bear  : TM_s>0, TM_m<0, TM_l<0
    Reversal_Bull     : TM_s>0, TM_m>0, TM_l<0
    Reversal_Bear     : TM_s<0, TM_m<0, TM_l>0
    Choppy_Bull       : TM_s>0, TM_m<0, TM_l>0
    Choppy_Bear       : TM_s<0, TM_m>0, TM_l<0
    Indecision        : otherwise
    """
    if periods is None:
        periods = [21, 64, 128]

    if len(periods) != 3:
        raise ValueError(
            "classify_n requires exactly 3 periods [short, mid, long]. "
            f"Got: {periods}"
        )

    ps, pm, pl = periods
    out = df.copy()

    # ── TM signals ────────────────────────────────────────────────────────
    tm_s = _safe_col(out, f"TM_{ps}")
    tm_m = _safe_col(out, f"TM_{pm}")
    tm_l = _safe_col(out, f"TM_{pl}")

    def pos(x):  return x >  tm_flat_thresh
    def neg(x):  return x < -tm_flat_thresh
    def flat(x): return x.abs() <= tm_flat_thresh

    all_up   = pos(tm_s) & pos(tm_m) & pos(tm_l)
    all_down = neg(tm_s) & neg(tm_m) & neg(tm_l)
    all_flat = flat(tm_s) & flat(tm_m) & flat(tm_l)

    retrace_bull  = neg(tm_s) & pos(tm_m) & pos(tm_l)
    retrace_bear  = pos(tm_s) & neg(tm_m) & neg(tm_l)
    reversal_bull = pos(tm_s) & pos(tm_m) & neg(tm_l)
    reversal_bear = neg(tm_s) & neg(tm_m) & pos(tm_l)
    choppy_bull   = pos(tm_s) & neg(tm_m) & pos(tm_l)
    choppy_bear   = neg(tm_s) & pos(tm_m) & neg(tm_l)

    # ── AC1 ───────────────────────────────────────────────────────────────
    ac1_s = _safe_col(out, f"AC1_{ps}")
    ac1_m = _safe_col(out, f"AC1_{pm}")
    ac1_reverting = (
        (ac1_s < ac1_reverting_thresh) &
        (ac1_m < ac1_reverting_thresh)
    )

    # ── Run stats ─────────────────────────────────────────────────────────
    dom_s     = _safe_col(out, f"DomRun_{ps}")
    max_pos_s = _safe_col(out, f"MaxRunPos_{ps}")
    max_neg_s = _safe_col(out, f"MaxRunNeg_{ps}")

    # ── RV spike ──────────────────────────────────────────────────────────
    rv_s      = _safe_col(out, f"RV_{ps}")
    rv_spike  = rv_s > rv_spike_pct * rv_s.rolling(pl).mean()

    # ── VMD ───────────────────────────────────────────────────────────────
    vmd_s = _safe_col(out, f"VMD_{ps}")

    # ── PVE and EMA slope ─────────────────────────────────────────────────
    pve_pct_s = _safe_col(out, f"PVE_pct_{ps}")
    pve_l     = _safe_col(out, f"PVE_{pl}")
    ema_sl_s  = _safe_col(out, f"EMA_slope_{ps}")

    overextended_up   = pve_pct_s > pve_pct_upper
    overextended_down = pve_pct_s < pve_pct_lower
    ema_flat          = ema_sl_s.abs() < ema_slope_flat

    # ── Composite regimes ─────────────────────────────────────────────────
    bull_exhaustion = (
        all_up & (dom_s < 0) &
        (ac1_s < ac1_reverting_thresh) & overextended_up
    )
    bear_exhaustion = (
        all_down & (dom_s > 0) &
        (ac1_s > -ac1_reverting_thresh) & overextended_down
    )
    exhaustion = bull_exhaustion | bear_exhaustion

    bull_capitul = reversal_bear & rv_spike & (max_neg_s > max_pos_s)
    bear_capitul = reversal_bull & rv_spike & (max_pos_s > max_neg_s)
    capitulation = bull_capitul | bear_capitul

    accumulation = (
        reversal_bull & (dom_s > 0) &
        (vmd_s == 0) & (pve_l >= -pve_ema_band)
    )
    distribution = (
        reversal_bear & (dom_s < 0) &
        (vmd_s == 0) & (pve_l <= pve_ema_band)
    )

    range_regime = all_flat & ac1_reverting & ema_flat

    # ── Priority order ────────────────────────────────────────────────────
    conditions = [
        exhaustion, capitulation,
        accumulation, distribution,
        range_regime,
        all_up, all_down,
        retrace_bull, retrace_bear,
        reversal_bull, reversal_bear,
        choppy_bull, choppy_bear,
    ]
    choices = [
        "TrendExhaustion", "TrendCapitulation",
        "Accumulation",    "Distribution",
        "Range",
        "TrendUp",         "TrendDown",
        "Retracement_Bull","Retracement_Bear",
        "Reversal_Bull",   "Reversal_Bear",
        "Choppy_Bull",     "Choppy_Bear",
    ]

    out["Regime"] = np.select(conditions, choices, default="Indecision")

    # Store TM direction for reference
    out["TM_dir_s"] = np.sign(tm_s)
    out["TM_dir_m"] = np.sign(tm_m)
    out["TM_dir_l"] = np.sign(tm_l)

    print(f"[INFO] Regime distribution:\n"
          f"{out['Regime'].value_counts().to_string()}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_all_features(df: pd.DataFrame,
                         periods: list = None,
                         doji_threshold: float = None,
                         doji_percentile: float = 20.0,
                         tm_flat_thresh: float = 0.10,
                         rv_spike_pct: float = 1.5,
                         ac1_reverting_thresh: float = -0.10,
                         pve_pct_upper: float = 90.0,
                         pve_pct_lower: float = 10.0,
                         pve_ema_band: float = 0.01,
                         ema_slope_flat: float = 0.001) -> pd.DataFrame:
    """
    Run the full deterministic feature engineering pipeline.

    Pipeline
    --------
    1.  candle_anatomy()   — Body, Wicks, Range, Ratios, CLV
    2.  compute_returns()  — r_day, r_night, r_total (log returns)
    3.  rolling_features() — R, RV, TM, ER, DC, AC, Run stats,
                              RVOL, VTC, VWR, VMD
    4.  price_vs_ema()     — EMA, PVE, PVE_pct, EMA_slope
    5.  conviction()       — Conviction ∈ {-1,-0.5,-0.25,0,+0.25,+0.5,+1}
    6.  momentum()         — MOM_{p} = R_{p} × mean(Conviction, p)
    7.  classify_single()  — single-bar pattern → Pattern_1
    8.  classify_double()  — two-bar pattern    → Pattern_2
    9.  classify_triple()  — three-bar pattern  → Pattern_3
    10. classify_n()       — regime             → Regime

    Parameters
    ----------
    df                   : OHLCV DataFrame [datetime, open, high, low,
                           close, volume]
    periods              : exactly 3 window sizes [short, mid, long]
                           (default [21, 64, 128])
    doji_threshold       : BodyRatio cutoff for doji (None = auto)
    doji_percentile      : percentile for auto doji threshold (20)
    tm_flat_thresh       : |TM| below this → flat in classify_n (0.10)
    rv_spike_pct         : RV spike multiplier for capitulation (1.5)
    ac1_reverting_thresh : AC1 threshold for mean-reversion (-0.10)
    pve_pct_upper        : PVE_pct upper overextension bound (90)
    pve_pct_lower        : PVE_pct lower overextension bound (10)
    pve_ema_band         : price tolerance around LT EMA (0.01 = 1%)
    ema_slope_flat       : |EMA_slope| below this → EMA flat (0.001)

    Returns
    -------
    pd.DataFrame with all original columns plus all computed features
    """
    if periods is None:
        periods = [21, 64, 128]

    if len(periods) != 3:
        raise ValueError(
            "periods must have exactly 3 elements [short, mid, long]. "
            f"Got: {periods}"
        )

    print(f"[INFO] Computing features | bars={len(df)} | periods={periods}")

    out = df.copy()

    print("[INFO] Step 1  — Candle anatomy...")
    out = candle_anatomy(out)

    print("[INFO] Step 2  — Returns...")
    out = compute_returns(out)

    print("[INFO] Step 3  — Rolling features...")
    out = rolling_features(out, periods)

    print("[INFO] Step 4  — Price vs EMA...")
    out = price_vs_ema(out, periods)

    print("[INFO] Step 5  — Conviction...")
    out = conviction(out, doji_threshold, doji_percentile)

    print("[INFO] Step 6  — Momentum...")
    out = momentum(out, periods)

    print("[INFO] Step 7  — Single-bar patterns...")
    out = classify_single(out)

    print("[INFO] Step 8  — Two-bar patterns...")
    out = classify_double(out)

    print("[INFO] Step 9  — Three-bar patterns...")
    out = classify_triple(out)

    print("[INFO] Step 10 — Regime classification...")
    out = classify_n(
        out,
        periods=periods,
        tm_flat_thresh=tm_flat_thresh,
        rv_spike_pct=rv_spike_pct,
        ac1_reverting_thresh=ac1_reverting_thresh,
        pve_pct_upper=pve_pct_upper,
        pve_pct_lower=pve_pct_lower,
        pve_ema_band=pve_ema_band,
        ema_slope_flat=ema_slope_flat
    )

    n_feat = len(out.columns) - len(df.columns)
    print(f"\n[INFO] Pipeline complete.")
    print(f"[INFO] Original columns : {len(df.columns)}")
    print(f"[INFO] Features added   : {n_feat}")
    print(f"[INFO] Total columns    : {len(out.columns)}")
    print(f"[INFO] Bars             : {len(out)}")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from schwabttk.price_history import load_stored

    df  = load_stored("M2K")
    out = compute_all_features(df, periods=[21, 64, 128])

    print("\n── Feature columns ──")
    print([c for c in out.columns if c not in df.columns])

    print("\n── Last 5 rows (key features) ──")
    cols = ["datetime", "r_day", "R_21", "TM_21", "ER_21",
            "Conviction", "MOM_21", "Pattern_1", "Regime"]
    print(out[cols].tail(5).to_string(index=False))

    print("\n── Single pattern distribution ──")
    print(out["Pattern_1"].value_counts())

    print("\n── Regime distribution ──")
    print(out["Regime"].value_counts())