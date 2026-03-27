"""
fomc_pricer.py
==============
FOMC meeting SOFR pricing table.

For each upcoming FOMC meeting, derives:
  - The correct overnight window to price (1D, 2D or 3D depending on
    whether the effective date falls before a weekend or holiday)
  - The implied SOFR for that window from the bootstrapped curve
  - The difference in bp from the current Fed target midpoint
  - The market-implied probability of a cut or hike (25bp increment)

Also provides a policy scenario generator (cartesian product of
cut/hike paths across meetings) and a daily rate series builder
for scenario analysis.

Settlement convention
---------------------
SOFR published on meeting day D reflects the prior overnight (D-1→D)
and carries NO meeting effect. The first overnight that prices in the
Fed decision accrues D → next_bd(D). This is what a 1-business-day
repo starting on meeting day D would settle at.

Usage
-----
    from crv_btstrpr import SOFRCurve
    from fomc_pricer import load_fed_meetings, fomc_pricing_table

    curve = SOFRCurve("marketinputs/sofr.xlsx")
    curve.bootstrap()

    table = fomc_pricing_table(
        curve,
        fed_mid   = 0.03625,       # current target midpoint
        fed_range = (0.035, 0.0375) # current target range
    )
    print(table)
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Optional, Dict, Any, Tuple

from crv_btstrpr import SOFRCurve


# ── 1. Load Fed Meeting Calendar ──────────────────────────────────────────────
def load_fed_meetings(
    t0: Optional[date] = None,
    calendar_path: Optional[str] = None,
    meetings: Optional[List[date]] = None,
) -> List[date]:
    """
    Return a sorted list of future FOMC meeting dates >= t0.

    Sources (in priority order):
      1. `meetings` — explicit list of date objects passed by caller
      2. CSV file at `calendar_path` (or default marketinputs/fed_meeting_calendar.csv)

    CSV format expected: columns Year, Month, Day (integer).
    """
    if t0 is None:
        t0 = date.today()
    else:
        t0 = pd.Timestamp(t0).date() if not isinstance(t0, date) else t0

    if meetings is not None:
        # manual override
        return sorted(
            [m if isinstance(m, date) else pd.Timestamp(m).date()
             for m in meetings
             if pd.Timestamp(m).date() >= t0]
        )

    if calendar_path is None:
        calendar_path = os.path.join(
            os.getcwd(), "marketinputs", "fed_meeting_calendar.csv"
        )

    df = pd.read_csv(calendar_path)
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"])
    ).dt.date

    return sorted(df.loc[df["date"] >= t0, "date"].tolist())


# ── 2. FOMC Pricing Table ─────────────────────────────────────────────────────
def fomc_pricing_table(
    curve: SOFRCurve,
    fed_mid: float,
    fed_range: Tuple[float, float],
    meetings: Optional[List[date]] = None,
    calendar_path: Optional[str] = None,
    step_bp: float = 25.0,
) -> pd.DataFrame:
    """
    For each upcoming FOMC meeting, compute the market-implied SOFR
    for the first overnight that prices in the Fed decision, expressed
    in bp vs the current target midpoint, with cut/hike probabilities.

    Parameters
    ----------
    curve       : bootstrapped SOFRCurve instance
    fed_mid     : current Fed target midpoint as decimal (e.g. 0.03625)
    fed_range   : (lower, upper) of current target range as decimals
    meetings    : optional manual list of meeting dates (overrides CSV)
    calendar_path: path to fed_meeting_calendar.csv
    step_bp     : increment for probability calculation (default 25bp)

    Returns
    -------
    DataFrame with columns:
        MeetingDate, EffectiveDate, WindowDays, ImpliedSOFR_%,
        Diff_bp, P(Cut)_%, P(Hike)_%,
        RateIfCut, RateIfHike, RateIfHold
    """
    mtg_dates = load_fed_meetings(
        t0=curve.today,
        calendar_path=calendar_path,
        meetings=meetings
    )

    step = step_bp / 10000.0  # decimal

    rows = []
    for mtg in mtg_dates:
        # First overnight embedding Fed decision: D → next_bd(D)
        eff_date  = curve.next_bd(mtg)
        eff_next  = curve.next_bd(eff_date)
        win_days  = (eff_date - mtg).days      # 1, 2, or 3 (Fri/holiday)

        # Implied SOFR for the overnight mtg → eff_date
        # = (DF(mtg) / DF(eff_date) - 1) * 360 / win_days
        t_mtg = curve.act360(curve.today, mtg)
        t_eff = curve.act360(curve.today, eff_date)
        df_mtg = curve.df(t_mtg)
        df_eff = curve.df(t_eff)
        implied_sofr = (df_mtg / df_eff - 1.0) * 360.0 / win_days

        # Difference from current midpoint in bp
        diff_bp = (implied_sofr - fed_mid) * 10000.0

        # Probability of a 25bp cut or hike
        # Assumes market prices a mixture of hold and one 25bp move
        # P(cut)  = max(0, (mid - implied) / step)  clipped to [0,1]
        # P(hike) = max(0, (implied - mid) / step)  clipped to [0,1]
        p_cut  = float(np.clip((fed_mid - implied_sofr) / step, 0.0, 1.0))
        p_hike = float(np.clip((implied_sofr - fed_mid) / step, 0.0, 1.0))

        rows.append({
            "MeetingDate":  mtg,
            "EffectiveDate": eff_date,
            "WindowDays":   win_days,
            "ImpliedSOFR_%": round(implied_sofr * 100, 4),
            "Diff_bp":      round(diff_bp, 2),
            "P(Cut)_%":     round(p_cut * 100, 1),
            "P(Hike)_%":    round(p_hike * 100, 1),
            "RateIfCut":    round((fed_mid - step) * 100, 4),
            "RateIfHike":   round((fed_mid + step) * 100, 4),
            "RateIfHold":   round(fed_mid * 100, 4),
        })

    return pd.DataFrame(rows)


# ── 3. Policy Scenario Generator ─────────────────────────────────────────────
def generate_policy_scenarios(
    initial_rate: float,
    t0: Optional[date] = None,
    step_size: float = 0.0025,
    max_cuts: int = 2,
    max_hikes: int = 2,
    max_moves_per_meeting: int = 1,
    rate_floor: Optional[float] = None,
    rate_cap: Optional[float] = None,
    calendar_path: Optional[str] = None,
    meetings: Optional[List[date]] = None,
) -> pd.DataFrame:
    """
    Generate all valid policy rate paths across future FOMC meetings
    as a cartesian product of per-meeting moves.

    Parameters
    ----------
    initial_rate          : starting Fed funds rate (decimal)
    step_size             : move size per increment (default 25bp)
    max_cuts / max_hikes  : total allowed across all meetings
    max_moves_per_meeting : max increments at any single meeting
    rate_floor / rate_cap : optional hard bounds on terminal rate
    """
    if t0 is None:
        t0 = date.today()

    mtg_dates = load_fed_meetings(t0=t0, calendar_path=calendar_path,
                                  meetings=meetings)
    n = len(mtg_dates)

    # Allowed move increments per meeting
    moves = [m for m in range(-max_moves_per_meeting, max_moves_per_meeting + 1)
             if abs(m) <= max_cuts or abs(m) <= max_hikes]

    rows = []
    for sid, path in enumerate(itertools.product(moves, repeat=n)):
        total_cuts  = -sum(m for m in path if m < 0)
        total_hikes =  sum(m for m in path if m > 0)

        if total_cuts > max_cuts or total_hikes > max_hikes:
            continue

        rate = initial_rate
        cut_dates  = []
        hike_dates = []
        for i, move in enumerate(path):
            if move < 0:
                rate += move * step_size
                cut_dates.append(mtg_dates[i].strftime("%Y-%m-%d"))
            elif move > 0:
                rate += move * step_size
                hike_dates.append(mtg_dates[i].strftime("%Y-%m-%d"))

        if rate_floor is not None and rate < rate_floor:
            continue
        if rate_cap is not None and rate > rate_cap:
            continue

        rows.append({
            "scenario_id":   sid,
            "total_cuts":    total_cuts,
            "total_hikes":   total_hikes,
            "cut_dates":     ", ".join(cut_dates) or None,
            "hike_dates":    ", ".join(hike_dates) or None,
            "terminal_rate": round(rate, 6),
        })

    return pd.DataFrame(rows)


# ── 4. Daily Rate Series Builder ──────────────────────────────────────────────
def build_daily_series_from_scenario(
    t0: date,
    horizon_days: int,
    initial_rate: float,
    cut_dates: List[date],
    cut_size: float,
    hike_dates: Optional[List[date]] = None,
    hike_size: Optional[float] = None,
    daycount_base: float = 360.0,
    repo_spread: float = 0.0,
) -> Dict[str, Any]:
    """
    Build a daily policy rate and discount factor series for a given scenario.

    The rate change on a meeting date takes effect on that day's overnight
    (meeting day D → next calendar day), consistent with SOFR T+1 publication.

    Parameters
    ----------
    t0            : start date (trade date)
    horizon_days  : number of calendar days to project
    initial_rate  : starting overnight rate (decimal)
    cut_dates     : dates on which a cut occurs
    cut_size      : size of each cut (decimal, e.g. 0.0025)
    hike_dates    : dates on which a hike occurs (optional)
    hike_size     : size of each hike (optional)
    daycount_base : 360 (Act/360, SOFR standard)
    repo_spread   : spread over policy rate (default 0)

    Returns
    -------
    dict with keys:
        calendar : list of dates (t0+1 .. t0+horizon_days)
        policy   : np.array of policy rates
        repo     : np.array of repo rates (policy + spread)
        dt       : 1 / daycount_base
        times    : year fractions from t0
        dfs      : cumulative discount factors from t0
    """
    dt       = 1.0 / daycount_base
    cal      = [t0 + timedelta(days=i) for i in range(1, horizon_days + 1)]
    n        = len(cal)
    policy   = np.empty(n, dtype=float)
    current  = float(initial_rate)

    cut_set  = {pd.Timestamp(d).date() for d in cut_dates}  if cut_dates  else set()
    hike_set = {pd.Timestamp(d).date() for d in hike_dates} if hike_dates else set()

    for i, d in enumerate(cal):
        d_date = d if isinstance(d, date) else d.date()
        if d_date in cut_set:
            current -= float(cut_size)
        if d_date in hike_set:
            current += float(hike_size or cut_size)
        policy[i] = current

    repo        = policy + float(repo_spread)
    log_dfs     = np.cumsum(np.log(1.0 / (1.0 + repo * dt)))
    dfs         = np.exp(log_dfs)
    times       = np.arange(1, n + 1) * dt

    return {
        "calendar": cal,
        "policy":   policy,
        "repo":     repo,
        "dt":       dt,
        "times":    times,
        "dfs":      dfs,
    }


# ── 5. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sofr_file = os.path.join(os.getcwd(), "marketinputs", "sofr.xlsx")
    today_date = pd.read_excel(
        sofr_file, sheet_name=0, header=0, usecols="F", nrows=1
    ).iloc[0, 0].date()

    curve = SOFRCurve(sofr_path=sofr_file, today=today_date, verbose=False)
    curve.bootstrap()

    # ── FOMC pricing table ────────────────────────────────────────────────────
    table = fomc_pricing_table(
        curve,
        fed_mid   = 0.03625,
        fed_range = (0.0350, 0.0375),
    )
    print("\n" + "=" * 70)
    print("FOMC Meeting SOFR Pricing Table")
    print("=" * 70)
    print(table.to_string(index=False))

    # ── Scenario generator ────────────────────────────────────────────────────
    scen = generate_policy_scenarios(
        initial_rate = 0.03625,
        t0           = today_date,
        max_cuts     = 2,
        max_hikes    = 0,
    )
    print("\n" + "=" * 70)
    print("Policy Scenarios")
    print("=" * 70)
    print(scen.to_string(index=False))