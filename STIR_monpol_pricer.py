import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from scipy.optimize import brentq
from matplotlib import pyplot as plt

# -------------------------------------------------------
# Load Fed Meeting Calendar
# -------------------------------------------------------
def load_fed_meetings(t0=None, calendar_path=None):
    """
    Loads fed_meeting_calendar.csv and returns
    list of future meeting dates >= t0 as pd.Timestamp objects.
    """

    if calendar_path is None:
        calendar_path = os.path.join(
            os.getcwd(),
            "marketinputs",
            "fed_meeting_calendar.csv"
        )

    df = pd.read_csv(calendar_path)

    # Ensure proper datetime construction
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"])
    ).dt.normalize()

    # Standardize t0 to Timestamp
    if t0 is None:
        t0 = pd.Timestamp.today().normalize()
    else:
        t0 = pd.Timestamp(t0).normalize()

    # Filter future meetings
    future_meetings = (
        df.loc[df["date"] >= t0, "date"]
        .sort_values()
        .reset_index(drop=True)
    )

    return future_meetings.tolist()

# -------------------------------------------------------
# Policy Rate Path Scenarios
# -------------------------------------------------------
def generate_policy_scenarios(
    initial_rate,
    t0=None,
    step_size=0.0025,
    max_cuts=2,
    max_hikes=2,
    max_moves_per_meeting=1,
    rate_floor=None,
    rate_cap=None,
    calendar_path=None
):
    """
    General policy step generator allowing cuts and hikes.

    Parameters
    ----------
    initial_rate : float
    step_size : float (default 25bp)
    max_cuts : int
    max_hikes : int
    max_moves_per_meeting : int (default 1)
    rate_floor : float or None
    rate_cap : float or None
    """
    if t0 is None:
        t0 = pd.Timestamp.today().normalize()
    else:
        t0 = pd.Timestamp(t0).normalize()

    meetings = load_fed_meetings(calendar_path)

    # Ensure meetings are Timestamps (defensive)
    meetings = [pd.Timestamp(m).normalize() for m in meetings]

    meetings = [m for m in meetings if m >= t0]
    n_meetings = len(meetings)

    # Allowed moves per meeting
    # Example: if max_moves_per_meeting=1
    # moves = [-1, 0, +1]
    moves = list(range(-max_cuts, max_hikes + 1))
    moves = [m for m in moves if abs(m) <= max_moves_per_meeting]

    scenarios = []
    scenario_id = 0

    # Cartesian product of meeting moves
    for path_moves in itertools.product(moves, repeat=n_meetings):

        total_cuts = -sum(m for m in path_moves if m < 0)
        total_hikes = sum(m for m in path_moves if m > 0)

        if total_cuts > max_cuts:
            continue
        if total_hikes > max_hikes:
            continue

        # Compute rate path
        rate = initial_rate
        cut_dates = []
        hike_dates = []

        for i, move in enumerate(path_moves):

            if move != 0:
                rate += move * step_size

                if move < 0:
                    cut_dates.append(meetings[i].strftime("%Y-%m-%d"))
                else:
                    hike_dates.append(meetings[i].strftime("%Y-%m-%d"))

        # Apply optional bounds
        if rate_floor is not None and rate < rate_floor:
            continue
        if rate_cap is not None and rate > rate_cap:
            continue

        scenarios.append({
            "scenario_id": scenario_id,
            "total_cuts": total_cuts,
            "total_hikes": total_hikes,
            "cut_dates": ", ".join(cut_dates) if cut_dates else None,
            "hike_dates": ", ".join(hike_dates) if hike_dates else None,
            "terminal_rate": rate
        })

        scenario_id += 1

    return pd.DataFrame(scenarios)

# ----------------------------
# Helper: build daily calendar and apply policy moves
# ----------------------------
def build_daily_series_from_scenario(
    t0: pd.Timestamp,
    horizon_days: int,
    initial_rate: float,
    cut_dates: List[date],
    cut_size: float,
    daycount_base: float = 364.0,
    repo_spread: float = 0.0
) -> Dict[str, Any]:
    """
    Returns:
      calendar: list of pd.Timestamp for each day (t0+1 ... t0+horizon_days)
      policy: np.array of policy rates per day
      repo: np.array of repo rates per day (policy + repo_spread)
      dt: 1/daycount_base
      times: year fractions corresponding to each day from t0
      dfs: cumulative discount factors to that day (from settlement t0)
    """
    # normalize t0
    if not isinstance(t0, pd.Timestamp):
        t0 = pd.Timestamp(t0).normalize()

    dt = 1.0 / float(daycount_base)
    # build calendar of dates (t0 + 1 ... t0 + horizon_days)
    cal = [ (t0 + pd.Timedelta(days=i)).normalize() for i in range(1, horizon_days + 1) ]
    n = len(cal)
    policy = np.empty(n, dtype=float)
    current = float(initial_rate)
    cut_set = set([pd.Timestamp(d).normalize() for d in cut_dates]) if cut_dates else set()

    for i, d in enumerate(cal):
        # apply cuts/hikes on the day they occur (change affects that day's overnight rate)
        if d in cut_set:
            current -= cut_size
        policy[i] = current

    repo = policy + float(repo_spread)

    # build cumulative discount factors DF_day = prod_{j=0..i} 1/(1 + repo_j * dt)
    one_plus_inv = 1.0 / (1.0 + repo * dt)
    # use log-cumsum to avoid numerical issues
    log_vals = np.log(one_plus_inv)
    log_cumsum = np.cumsum(log_vals)
    dfs = np.exp(log_cumsum)
    times = np.arange(1, n+1) * dt  # years from t0

    return {"calendar": cal, "policy": policy, "repo": repo, "dt": dt, "times": times, "dfs": dfs}

# ----------------------------
# Helper: accumulation factor from repo series (for CMT2Y)
# ----------------------------
def accumulation_from_repo(repo: np.ndarray, dt: float) -> float:
    # accumulation = prod(1 + repo_i * dt)
    log_acc = np.log(1.0 + repo * dt).sum()
    return float(np.exp(log_acc))

def accumulation_to_cmt2y(accum: float) -> float:
    # map 2-year accumulation to semiannual-compounded annual yield
    return 2.0 * (accum ** (1.0 / 4.0) - 1.0)

# ----------------------------
# Helper: interpolate DF at year fraction t using times, dfs arrays
# ----------------------------
def df_at_t(t: float, times: np.ndarray, dfs: np.ndarray) -> float:
    if t <= 0:
        return 1.0
    if t >= times[-1]:
        return float(dfs[-1])
    return float(np.interp(t, times, dfs))

# ----------------------------
# Helper: price CTD bond using daily dfs
# ----------------------------
def price_ctd_from_daily_dfs(
    settlement: pd.Timestamp,
    coupon_rate: float,
    coupon_dates: List[date],
    times: np.ndarray,
    dfs: np.ndarray,
    face: float = 100.0,
    freq: int = 2
) -> Dict[str, float]:
    """
    coupon_dates: list of python.date or pd.Timestamp future coupon dates > settlement
    We use ACT/365 to compute t (year fraction) for coupon timing (keeps coupon schedule convention).
    """
    coupon_payment = coupon_rate / freq * face
    pv = 0.0
    # ensure coupon_dates sorted and are pd.Timestamp
    cds = [pd.Timestamp(d).normalize() for d in sorted(coupon_dates)]
    for i, d in enumerate(cds):
        t = (d - pd.Timestamp(settlement)).days / 365.0  # ACT/365 for coupon timing
        df = df_at_t(t, times, dfs)
        if i == len(cds) - 1:
            pv += (coupon_payment + face) * df
        else:
            pv += coupon_payment * df

    # --- Find next coupon ---
    next_coupon = None
    for d in cds:
        if d > pd.Timestamp(settlement):
            next_coupon = d
            break

    if next_coupon is None:
        raise ValueError("Settlement beyond final coupon.")

    # --- Find previous coupon ---
    # For regular UST semiannual schedule:
    prev_coupon = next_coupon - pd.DateOffset(months=6)
    # --- Accrual calculation ---
    accr_days = (pd.Timestamp(settlement) - prev_coupon).days
    coupon_period_days = (next_coupon - prev_coupon).days

    accr_fraction = accr_days / coupon_period_days
    accrued = coupon_payment * accr_fraction

    dirty = pv
    clean = dirty - accrued
    return {"dirty": dirty, "clean": clean, "accrued": accrued}

# ----------------------------
# Helper: solve YTM from dirty price (semiannual compounding)
# ----------------------------
def solve_ytm_from_dirty(dirty_price: float, coupon_rate: float, coupon_dates: List[date], settlement: date, face: float = 100.0, freq: int = 2) -> float:
    c = coupon_rate / freq * face
    periods = len(coupon_dates)

    def price_from_y(y):
        total = 0.0
        for i in range(1, periods + 1):
            total += c / (1.0 + y / freq) ** i
        total += face / (1.0 + y / freq) ** periods
        return total

    try:
        ytm = brentq(lambda y: price_from_y(y) - dirty_price, -0.5, 1.0, maxiter=200)
    except Exception:
        ytm = np.nan
    return float(ytm)

# ----------------------------
# Main: connect scenarios -> pricing
# ----------------------------
def price_policy_scenarios(
    scenario_df: pd.DataFrame,
    initial_rate: float,
    t0: Optional[pd.Timestamp] = None,
    daycount_base: float = 364.0,
    repo_spread: float = 0.0,
    cut_size: float = 0.0025,
    horizon_years: float = 2.0,
    # CTD bond specifics (optional)
    ctd_coupon: Optional[float] = None,
    ctd_maturity: Optional[date] = None,
    settlement: Optional[date] = None,
    face: float = 100.0,
    freq: int = 2
) -> pd.DataFrame:
    """
    For each scenario (expects columns: cut_dates (ISO string) OR cut_indices & meetings),
    compute CMT2Y and optionally CTD dirty/clean/ytm.
    Returns scenario_df augmented with numeric columns.
    """
    # normalize t0
    if t0 is None:
        t0 = pd.Timestamp.today().normalize()
    else:
        t0 = pd.Timestamp(t0).normalize()

    # horizon in days according to chosen daycount_base
    horizon_days = int(round(horizon_years * float(daycount_base)))

    out = []
    # Acquire full meetings list to map cut_indices (if needed)
    meetings_all = load_fed_meetings()  # returns list of Timestamps
    # filter future meetings relative to t0
    meetings_future = [m for m in meetings_all if m >= t0]

    for _, row in scenario_df.iterrows():
        # parse cuts into list of dates
        if pd.notnull(row.get("cut_dates")):
            # cut_dates expected "YYYY-MM-DD, YYYY-MM-DD" or None
            cut_dates = [pd.Timestamp(s.strip()).normalize() for s in str(row["cut_dates"]).split(",")] if row["cut_dates"] else []
        else:
            cut_indices = row.get("cut_indices", ())
            cut_dates = [meetings_future[i].normalize() for i in cut_indices] if cut_indices else []

        # keep only cuts within horizon
        cut_dates = [d for d in cut_dates if 0 < (d - t0).days <= horizon_days]

        ds = build_daily_series_from_scenario(
            t0=t0,
            horizon_days=horizon_days,
            initial_rate=initial_rate,
            cut_dates=cut_dates,
            cut_size=cut_size,
            daycount_base=daycount_base,
            repo_spread=repo_spread
        )

        accum = accumulation_from_repo(ds["repo"], ds["dt"])
        cmt2y = accumulation_to_cmt2y(accum)

        out_row = row.to_dict()
        out_row.update({
            "accum_factor": accum,
            "cmt2y": float(cmt2y)
        })

        # If CTD inputs provided, price CTD using daily dfs and solve for YTM
        if ctd_coupon is not None and ctd_maturity is not None and settlement is not None:
            # Build coupon schedule (future coupon dates) - simple semiannual on standard months assumed
            # You should provide exact coupon dates for precision; here we create an expected schedule:
            coupon_dates = []
            s = pd.Timestamp(settlement).normalize()
            mat = pd.Timestamp(ctd_maturity).normalize()
            # generate semiannual dates between settlement and maturity (naive generator)
            cur = mat
            # build backward then reverse
            while cur > s:
                coupon_dates.append(cur)
                # step back ~6 months
                cur = (cur - pd.DateOffset(months=6)).normalize()
            coupon_dates = sorted(coupon_dates)

            prices = price_ctd_from_daily_dfs(
                settlement=pd.Timestamp(settlement).normalize(),
                coupon_rate=ctd_coupon,
                coupon_dates=coupon_dates,
                times=ds["times"],
                dfs=ds["dfs"],
                face=face,
                freq=freq
            )
            out_row.update({
                "ctd_dirty": float(prices["dirty"]),
                "ctd_clean": float(prices["clean"]),
                "ctd_accrued": float(prices["accrued"]),
            })
            ytm = solve_ytm_from_dirty(prices["dirty"], ctd_coupon, coupon_dates, pd.Timestamp(settlement).normalize(), face=face, freq=freq)
            out_row.update({
                "ctd_ytm": float(ytm)
            })

        out.append(out_row)

    return pd.DataFrame(out)

# ----------------------------
# Example usage:
# ----------------------------
scen = generate_policy_scenarios(initial_rate=0.03625, max_cuts=2, max_hikes=0)
priced = price_policy_scenarios(scen, initial_rate=0.03625,
                                t0=pd.Timestamp("2026-02-23"),
                                daycount_base=364.0, repo_spread=0.0,
                                cut_size=0.0025,
                                ctd_coupon=0.035,
                                ctd_maturity=date(2028,1,31),
                                settlement=date(2026,2,12))
print(priced[["scenario_id","total_cuts","cut_dates","terminal_rate","cmt2y","ctd_ytm"]])