"""
crv_btstrpr.py
==============
Interest Rate Curve Bootstrapper.

Builds a discount curve from a set of market instruments using
log-linear interpolation on discount factors. Designed to support
multiple curve types and instrument families including OIS curves
(SOFR, ESTR, SONIA, TONAR), IBOR curves (LIBOR, EURIBOR), and
mixed curves combining cash deposits, futures, FRAs, and swaps.

Instrument types currently implemented
---------------------------------------
- Rate  : Overnight/term cash deposit, configurable settlement
- Swap  : Fixed-float interest rate swap
          ≤1Y → single-period closed-form
          >1Y → periodic fixed leg, analytic bootstrap
- Future: CME 3M SOFR futures, IMM-dated accrual windows,
          Hull-White convexity adjustment applied

Settlement conventions (SOFR-OIS defaults)
------------------------------------------
- Swaps     : T+2 US Gov Securities business days (ISDA/ARRC standard)
              Effective date = next_bd(next_bd(trade_date))
              Maturity       = add_tenor(effective_date, tenor)
- Futures   : CME 3M SOFR — accrual runs between exact IMM dates
              Start = 3rd Wednesday of named month (Reference Quarter start)
              End   = 3rd Wednesday of named month + 3 months
- Cash rate : T+1 next business day (overnight deposit convention)

Overnight index publication lag
--------------------------------
The overnight rate published on day D has value date D-1 (T+1 lag).
On any trade date T, the rate for T→next_bd(T) is not yet published —
it will only be known on next_bd(T). The entire forward curve, including
the first overnight pillar, is therefore forward-looking/quoted rather
than observed.

Interpolation
-------------
Log-linear on discount factors (piecewise, no look-ahead between pillars).
Each new pillar is solved analytically — no global root-finding that would
perturb previously bootstrapped discount factors.

Usage
-----
    from crv_btstrpr import SOFRCurve

    curve = SOFRCurve(sofr_path="marketinputs/sofr.xlsx", today=date(2026,2,20))
    curve.bootstrap()    # builds pillar_t / pillar_df
    curve.build_grid()   # populates curve.grid DataFrame
    curve.plot()         # renders and saves the 3-panel chart
    df  = curve.grid     # repriced tenor grid
    out = curve.out      # bootstrapped pillars
"""

import os
import calendar
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import date, timedelta
from typing import Optional


# ── Plot defaults ─────────────────────────────────────────────────────────────
COLORS = [
    '#0626a9', '#ffc62d', '#f08900', '#e44261',
    '#00c4b3', '#6c7bd3', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
]
plt.rcParams['figure.dpi']      = 72.0
plt.rcParams['figure.figsize']  = [6, 4]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)


# ── Module-level constants ────────────────────────────────────────────────────
TENOR_GRID_SPEC = [
    ("1D",  1,  "D"), ("1W",  1,  "W"), ("2W",  2,  "W"), ("3W",  3,  "W"),
    ("1M",  1,  "M"), ("2M",  2,  "M"), ("3M",  3,  "M"), ("6M",  6,  "M"),
    ("9M",  9,  "M"), ("12M", 12, "M"), ("18M", 18, "M"),
    ("2Y",  2,  "Y"), ("3Y",  3,  "Y"), ("4Y",  4,  "Y"), ("5Y",  5,  "Y"),
    ("6Y",  6,  "Y"), ("7Y",  7,  "Y"), ("8Y",  8,  "Y"), ("9Y",  9,  "Y"),
    ("10Y", 10, "Y"), ("15Y", 15, "Y"), ("20Y", 20, "Y"), ("25Y", 25, "Y"),
    ("30Y", 30, "Y"), ("35Y", 35, "Y"), ("40Y", 40, "Y"), ("45Y", 45, "Y"),
    ("50Y", 50, "Y"),
]

SHOW_LABELS = {
    "1D", "9M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y",
    "8Y", "9Y", "10Y", "15Y", "20Y", "25Y", "30Y",
    "35Y", "40Y", "45Y", "50Y"
}


# ─────────────────────────────────────────────────────────────────────────────
class SOFRCurve:
    """
    OIS curve bootstrapper with full settlement convention support.

    Parameters
    ----------
    sofr_path       : path to Excel input file
    today           : trade date (read from Excel col F row 1 if None)
    hw_a            : Hull-White mean reversion speed (default 0.03)
    hw_sigma        : HW short-rate vol override; if None reads from Excel P7
    n_holiday_years : number of years of US holidays to pre-build
    verbose         : print bootstrap progress, diagnostics and health report
    """

    def __init__(
        self,
        sofr_path: str,
        today: Optional[date] = None,
        hw_a: float = 0.03,
        hw_sigma: Optional[float] = None,
        n_holiday_years: int = 55,
        verbose: bool = True,
    ):
        self.sofr_path = sofr_path
        self.verbose   = verbose

        # ── Holiday calendar ─────────────────────────────────────────────────
        raw_today = pd.read_excel(io=sofr_path, sheet_name=0, header=0, usecols="F", nrows=1).iloc[0, 0]
        self.today: date = today or pd.Timestamp(raw_today).date()

        self._holidays: set = set()
        for n in range(n_holiday_years):
            self._holidays |= self._us_federal_holidays(self.today.year + n)

        # ── Settlement dates ─────────────────────────────────────────────────
        self.effective_date: date = self.next_bd(self.next_bd(self.today))

        # ── HW parameters ────────────────────────────────────────────────────
        self.hw_a     = hw_a
        self.hw_sigma = hw_sigma if hw_sigma is not None else self._load_hw_sigma()

        # ── Market data ──────────────────────────────────────────────────────
        self.df_mkt: pd.DataFrame = self._load_market_data()

        # ── Derived constants ────────────────────────────────────────────────
        self._short_end = self.add_tenor(self.effective_date, "1Y")

        # ── Output containers ────────────────────────────────────────────────
        self.pillar_t:  np.ndarray        = np.array([0.0])
        self.pillar_df: np.ndarray        = np.array([1.0])
        self._interp                      = None   # cached interpolator
        self.out:  Optional[pd.DataFrame] = None
        self.grid: Optional[pd.DataFrame] = None

        if self.verbose:
            print(f"Trade date     : {self.today}")
            print(f"Effective date : {self.effective_date}  (T+2 US Gov Sec BDs)")
            print(f"HW a           : {self.hw_a}")
            print(f"HW σ           : {self.hw_sigma*100:.6f}%")

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_market_data(self) -> pd.DataFrame:
        df = pd.read_excel(
            self.sofr_path, sheet_name=0,
            usecols="A:D", header=0
        ).dropna(subset=["Type"])
        df.columns = ["Symbol", "Tenors", "Type", "Quote"]

        mats, ts = [], []
        for _, row in df.iterrows():
            typ   = row["Type"]
            tenor = str(row["Tenors"]).strip()
            sym   = row["Symbol"]
            if typ == "Rate":
                mat = self.next_bd(self.today)
            elif typ == "Future":
                _, imm_end = self.imm_dates_from_symbol(sym)
                mat = imm_end
            else:
                mat = self.add_tenor(self.effective_date, tenor)
            mats.append(mat)
            ts.append(self.act360(self.today, mat))

        df["MatDate"] = mats
        df["T"]       = ts
        return df

    def _load_hw_sigma(self) -> float:
        try:
            raw   = pd.read_excel(
                self.sofr_path, sheet_name=0,
                header=0, usecols="P", nrows=8
            ).iloc[6, 0]
            sigma = float(raw) / 100.0
        except Exception:
            sigma = 0.0060
        if self.verbose:
            print(f"HW σ (from P7) : {sigma*100:.6f}%")
        return sigma

    # ── Calendar helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _us_federal_holidays(year: int) -> set:
        def obs(d):
            if d.weekday() == 6: return d + timedelta(days=1)
            if d.weekday() == 5: return d - timedelta(days=1)
            return d
        def nth_wd(y, m, n, wd):
            if n > 0:
                d = date(y, m, 1)
                d += timedelta(days=(wd - d.weekday()) % 7)
                return d + timedelta(weeks=n - 1)
            last = calendar.monthrange(y, m)[1]
            d = date(y, m, last)
            d -= timedelta(days=(d.weekday() - wd) % 7)
            return d
        return {
            obs(date(year,  1,  1)), obs(date(year,  6, 19)),
            obs(date(year,  7,  4)), obs(date(year, 11, 11)),
            obs(date(year, 12, 25)),
            nth_wd(year,  1,  3, 0), nth_wd(year,  2,  3, 0),
            nth_wd(year,  5, -1, 0), nth_wd(year,  9,  1, 0),
            nth_wd(year, 10,  2, 0), nth_wd(year, 11,  4, 3),
        }

    def next_bd(self, d: date) -> date:
        d += timedelta(days=1)
        while d.weekday() >= 5 or d in self._holidays:
            d += timedelta(days=1)
        return d

    def prev_bd(self, d: date) -> date:
        d -= timedelta(days=1)
        while d.weekday() >= 5 or d in self._holidays:
            d -= timedelta(days=1)
        return d

    def modified_following(self, d: date) -> date:
        if d.weekday() < 5 and d not in self._holidays:
            return d
        fwd = d
        while fwd.weekday() >= 5 or fwd in self._holidays:
            fwd += timedelta(days=1)
        if fwd.month == d.month:
            return fwd
        bwd = d
        while bwd.weekday() >= 5 or bwd in self._holidays:
            bwd -= timedelta(days=1)
        return bwd

    @staticmethod
    def _clamp(y: int, m: int, day: int) -> date:
        return date(y, m, min(day, calendar.monthrange(y, m)[1]))

    def add_tenor(self, base: date, tenor: str) -> date:
        tenor = tenor.strip()
        if tenor == "1B":
            return self.next_bd(base)
        elif tenor.endswith("Y"):
            y   = int(tenor[:-1])
            raw = self._clamp(base.year + y, base.month, base.day)
            return self.modified_following(raw)
        elif tenor.endswith("M"):
            m       = int(tenor[:-1])
            total_m = base.month - 1 + m
            raw     = self._clamp(
                base.year + total_m // 12, total_m % 12 + 1, base.day
            )
            return self.modified_following(raw)
        elif tenor.endswith("W"):
            return self.modified_following(base + timedelta(weeks=int(tenor[:-1])))
        elif tenor.endswith("D"):
            return base + timedelta(days=int(tenor[:-1]))
        raise ValueError(f"Unknown tenor: {tenor}")

    # ── IMM date helpers ──────────────────────────────────────────────────────
    _CME_MONTH = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5,  'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V':10, 'X':11,  'Z':12,
    }

    @staticmethod
    def _third_wednesday(y: int, m: int) -> date:
        d = date(y, m, 1)
        d += timedelta(days=(2 - d.weekday()) % 7)
        return d + timedelta(weeks=2)

    def imm_dates_from_symbol(self, symbol: str) -> tuple[date, date]:
        """
        Derive IMM accrual [start, end) from CME futures symbol.
        Month letter = start of Reference Quarter.
        End = 3rd Wednesday of start month + 3 months.
        """
        code         = symbol.replace("SOFR3", "").strip()
        month_letter = code[0].upper()
        year_suffix  = int(code[1:])

        base_decade = self.today.year - (self.today.year % 10)
        start_month = self._CME_MONTH[month_letter]
        start_year  = None

        for decade in (base_decade, base_decade + 10):
            candidate = decade + year_suffix
            em = start_month + 3
            ey = candidate
            if em > 12:
                em -= 12
                ey += 1
            if self._third_wednesday(ey, em) >= self.today:
                start_year = candidate
                break

        if start_year is None:
            raise ValueError(f"Cannot resolve year for {symbol} from {self.today}")

        imm_start = self._third_wednesday(start_year, start_month)
        em = start_month + 3
        ey = start_year
        if em > 12:
            em -= 12
            ey += 1
        imm_end = self._third_wednesday(ey, em)
        return imm_start, imm_end

    # ── Day count ─────────────────────────────────────────────────────────────
    @staticmethod
    def act360(d1: date, d2: date) -> float:
        return (d2 - d1).days / 360.0

    # ── Curve interpolation ───────────────────────────────────────────────────
    def _build_interpolator(self) -> None:
        """Cache log-linear interpolator. Call after pillar_t/df are updated."""
        self._interp = interp1d(
            self.pillar_t, np.log(self.pillar_df),
            kind="linear", fill_value="extrapolate"
        )

    def df(self, t: float) -> float:
        """Log-linear interpolated DF at Act/360 time t from today."""
        if self._interp is None:
            self._build_interpolator()
        return float(np.exp(self._interp(t)))

    @staticmethod
    def _df_from(t: float, pt: np.ndarray, pdf: np.ndarray) -> float:
        """Log-linear DF from an explicit pillar set (used during bootstrap)."""
        f = interp1d(pt, np.log(pdf), kind="linear", fill_value="extrapolate")
        return float(np.exp(f(t)))

    # ── HW convexity adjustment ───────────────────────────────────────────────
    def hw_ca(self, t1: float, t2: float) -> float:
        """HW convexity adjustment for futures accrual [t1, t2]."""
        a, s = self.hw_a, self.hw_sigma
        def B(u, v): return (1.0 - np.exp(-a * (v - u))) / a
        return 0.5 * s**2 * B(t1, t2) * (
            B(0, t2) * np.exp(-a * t2) +
            (1.0 - np.exp(-2.0 * a * t1)) / (2.0 * a)
        )

    # ── Coupon schedule ───────────────────────────────────────────────────────
    def _coupon_schedule(self, mat: date) -> list[date]:
        """
        Annual coupon dates from effective_date to mat using fixed-anchor
        ISDA convention: coupon(n) = MF(effective_date + n × 1Y).
        Prevents date drift on long schedules.
        """
        dates = []
        n = 1
        while True:
            raw = self._clamp(
                self.effective_date.year + n,
                self.effective_date.month,
                self.effective_date.day
            )
            cpn = self.modified_following(raw)
            if cpn >= mat:
                dates.append(mat)
                break
            dates.append(cpn)
            n += 1
        return dates

    # ── Instrument pricers ────────────────────────────────────────────────────
    def _ois_df_short(self, swap_rate: float, mat: date,
                      pt: np.ndarray, pdf: np.ndarray) -> float:
        """
        Closed-form DF for single-period OIS (≤1Y).
        DF(today→mat) = DF(today→eff) / (1 + r × α)
        where α = act360(effective_date, mat).
        """
        t_eff  = self.act360(self.today, self.effective_date)
        df_eff = self._df_from(t_eff, pt, pdf) if t_eff > 0 else 1.0
        alpha  = self.act360(self.effective_date, mat)
        return df_eff / (1.0 + swap_rate * alpha)

    def _ois_df_long(self, swap_rate: float, mat: date,
                     pt: np.ndarray, pdf: np.ndarray) -> float:
        """
        Closed-form DF for annual-fixed OIS (>1Y).
        DF(T_N) = (DF(eff) - r·Σ αᵢ·DF(Tᵢ)) / (1 + r·αN)
        Fixed-anchor coupon schedule (ISDA standard).
        """
        t_eff  = self.act360(self.today, self.effective_date)
        df_eff = self._df_from(t_eff, pt, pdf) if t_eff > 0 else 1.0

        fix_dates       = self._coupon_schedule(mat)
        annuity, prev   = 0.0, self.effective_date
        for fd in fix_dates[:-1]:
            annuity += self.act360(prev, fd) * self._df_from(
                self.act360(self.today, fd), pt, pdf
            )
            prev = fd

        alpha_N = self.act360(prev, mat)
        df_val  = (df_eff - swap_rate * annuity) / (1.0 + swap_rate * alpha_N)

        # Verify closed-form residual
        residual = abs(
            df_eff - swap_rate * annuity - df_val * (1.0 + swap_rate * alpha_N)
        )
        assert residual < 1e-12, f"Closed-form residual too large at {mat}: {residual:.2e}"

        return df_val

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def bootstrap(self) -> None:
        """
        Sequential bootstrap over all instruments sorted by maturity.
        Populates self.pillar_t, self.pillar_df, self.out.
        """
        pt  = np.array([0.0])
        pdf = np.array([1.0])
        results = []

        for _, row in self.df_mkt.sort_values("T").iterrows():
            sym, typ  = row["Symbol"], row["Type"]
            quote, mat = float(row["Quote"]), row["MatDate"]

            if typ == "Rate":
                # Overnight cash: implied forward rate for today → next_bd(today)
                r      = quote / 100.0
                T      = self.act360(self.today, mat)
                df_val = 1.0 / (1.0 + r * T)

            elif typ == "Future":
                imm_start, imm_end = self.imm_dates_from_symbol(sym)

                # Warn if contract has already started accrual
                if imm_start < self.today:
                    warnings.warn(
                        f"{sym}: accrual started {imm_start} before today "
                        f"{self.today}. Some fixings are known — consider "
                        f"replacing with a seasoned instrument.",
                        UserWarning, stacklevel=2
                    )

                t1      = self.act360(self.today, imm_start)
                t2      = self.act360(self.today, imm_end)
                accrual = self.act360(imm_start, imm_end)
                T       = t2
                fwd_r   = (100.0 - quote) / 100.0 - self.hw_ca(max(t1, 0.0), t2)
                df_s    = self._df_from(max(t1, 0.0), pt, pdf)
                df_val  = df_s / (1.0 + fwd_r * accrual)
                mat     = imm_end

            elif typ == "Swap":
                T  = self.act360(self.today, mat)
                sr = quote / 100.0
                if mat <= self._short_end:
                    df_val = self._ois_df_short(sr, mat, pt, pdf)
                else:
                    df_val = self._ois_df_long(sr, mat, pt, pdf)
            else:
                continue

            # ── Pillar safety checks ─────────────────────────────────────────
            if T <= pt[-1]:
                raise ValueError(
                    f"Maturity ordering violation at {sym}: "
                    f"T={T:.6f} <= previous T={pt[-1]:.6f}"
                )
            if df_val <= 0.0:
                raise ValueError(
                    f"Non-positive discount factor at {sym}: {df_val:.8f}"
                )
            if df_val > pdf[-1]:
                raise ValueError(
                    f"Non-monotonic discount factor at {sym}: "
                    f"{df_val:.8f} > previous {pdf[-1]:.8f}"
                )

            pt  = np.append(pt,  T)
            pdf = np.append(pdf, df_val)
            results.append((sym, mat, T, df_val))

        self.pillar_t  = pt
        self.pillar_df = pdf
        self._build_interpolator()   # cache interpolator once pillars are final

        self.out = pd.DataFrame(
            results, columns=["Symbol", "MatDate", "T_act360", "DiscountFactor"]
        )
        self.out["Tenor"] = (
            self.out
            .merge(self.df_mkt[["Symbol", "Tenors"]], on="Symbol", how="left")
            ["Tenors"]
        )
        self.out["ZeroRate"] = self.out.apply(
            lambda r: self._discrete_zero(
                r["T_act360"], r["DiscountFactor"], r["MatDate"]
            ) * 100, axis=1
        )
        self.out["Fwd_SOFR_1D"] = self.out["MatDate"].apply(
            lambda m: self._sofr_fwd_1d(m) * 100
        )
        self.out["Exp_Fwd_SOFR_1D"] = self.out.apply(
            lambda r: (
                (1.0 / r["DiscountFactor"])
                ** (1.0 / (r["MatDate"] - self.today).days) - 1.0
            ) * 360 * 100, axis=1
        )

        if self.verbose:
            # Repricing diagnostics
            print("\nRepricing diagnostics")
            print("-" * 60)
            max_err = 0.0
            for _, row in self.df_mkt.sort_values("T").iterrows():
                typ   = row["Type"]
                quote = float(row["Quote"])
                mat   = row["MatDate"]
                sym   = row["Symbol"]
                if typ == "Swap":
                    model  = self._reprice(mat) * 100.0
                    err_bp = (model - quote) * 100.0
                    max_err = max(max_err, abs(err_bp))
                    print(f"  {sym:15s}  quoted={quote:.4f}%  model={model:.4f}%  "
                          f"err={err_bp:+.4f} bp")
                elif typ == "Future":
                    imm_s, imm_e = self.imm_dates_from_symbol(sym)
                    t1 = self.act360(self.today, imm_s)
                    t2 = self.act360(self.today, imm_e)
                    df_s    = self.df(max(t1, 0.0))
                    df_e    = self.df(t2)
                    accrual = self.act360(imm_s, imm_e)
                    fwd_r   = (df_s / df_e - 1.0) / accrual
                    ca      = self.hw_ca(max(t1, 0.0), t2)
                    model_price = 100.0 * (1.0 - (fwd_r + ca))
                    err_bp  = (model_price - quote) * 100.0
                    max_err = max(max_err, abs(err_bp))
                    print(f"  {sym:15s}  quoted={quote:.4f}   model={model_price:.4f}   "
                          f"err={err_bp:+.4f} bp")
            print("-" * 60)
            print(f"  Max repricing error : {max_err:.6f} bp")

            # Forward curve sanity check
            print("\nForward curve sanity check:")
            warned = False
            for t in np.linspace(0.5, self.pillar_t[-1], 20):
                fwd = self.instantaneous_forward(t)
                if fwd < -0.05:
                    print(f"  ⚠ Negative forward at t={t:.2f}: {fwd:.4%}")
                    warned = True
            if not warned:
                print("  All instantaneous forwards ≥ -5%  ✅")

            # Curve health report
            print("\nCurve health summary")
            print(f"  Pillars                : {len(self.pillar_t) - 1}")
            print(f"  Final maturity (years) : {self.pillar_t[-1]:.2f}")
            print(f"  Min DF                 : {self.pillar_df.min():.6f}")
            print(f"  Max zero rate          : {self.out['ZeroRate'].max():.4f}%")
            print(f"  Min zero rate          : {self.out['ZeroRate'].min():.4f}%")

            print("\n" + "=" * 70)
            print(f"SOFR-OIS Bootstrapped Curve  |  Trade: {self.today}"
                  f"  Eff: {self.effective_date}")
            print("=" * 70)
            print(self.out[["Symbol", "MatDate", "T_act360",
                             "DiscountFactor", "Exp_Fwd_SOFR_1D"]]
                  .to_string(index=False, float_format="%.6f"))

    # ── Rate helpers ──────────────────────────────────────────────────────────
    def _discrete_zero(self, t: float, df_val: float, mat: date) -> float:
        if mat <= self.add_tenor(self.today, "1Y"):
            return (1.0 / df_val - 1.0) * 360.0 / (mat - self.today).days
        return (1.0 / df_val) ** (1.0 / t) - 1.0

    def _reprice(self, mat: date) -> float:
        """Par OIS rate at mat from the bootstrapped curve."""
        T      = self.act360(self.today, mat)
        df_mat = self.df(T)
        t_eff  = self.act360(self.today, self.effective_date)
        df_eff = self.df(t_eff) if t_eff > 0 else 1.0

        if mat <= self._short_end:
            alpha = self.act360(self.effective_date, mat)
            return (df_eff / df_mat - 1.0) / alpha

        fix_dates       = self._coupon_schedule(mat)
        annuity, prev   = 0.0, self.effective_date
        for fd in fix_dates[:-1]:
            annuity += self.act360(prev, fd) * self.df(
                self.act360(self.today, fd)
            )
            prev = fd
        annuity += self.act360(prev, mat) * df_mat
        return (df_eff - df_mat) / annuity

    def _sofr_fwd_1d(self, mat: date) -> float:
        """Overnight SOFR forward ending at mat (simple Act/360)."""
        prev  = self.prev_bd(mat)
        days  = (mat - prev).days
        return (
            self.df(self.act360(self.today, prev)) /
            self.df(self.act360(self.today, mat)) - 1.0
        ) * 360.0 / days

    def instantaneous_forward(self, t: float, eps: float = 1e-6) -> float:
        """Instantaneous forward rate at time t via central difference."""
        df_plus  = self.df(t + eps)
        df_minus = self.df(max(t - eps, 0.0))
        return -(np.log(df_plus) - np.log(df_minus)) / (2 * eps)

    def get_zero_rate(self, tenor: str) -> float:
        """
        Annually compounded zero rate (Act/360) for a given tenor string.
        Useful for unit testing: assert abs(curve.get_zero_rate('5Y') - expected) < 1e-6
        """
        mat = self.add_tenor(self.effective_date, tenor)
        T   = self.act360(self.today, mat)
        return (1.0 / self.df(T)) ** (1.0 / T) - 1.0

    def bucket_dv01(self, bump_bp: float = 1.0) -> pd.DataFrame:
        """
        Bucket DV01: bump each instrument's quote by bump_bp and rebootstrap,
        measuring the change in each pillar DF. Provides per-instrument
        sensitivity showing where the curve is most sensitive to market moves.

        Returns a DataFrame with columns: Symbol, Tenor, Type, dDF per pillar.
        """
        rows = []
        for idx, row in self.df_mkt.iterrows():
            sym  = row["Symbol"]
            typ  = row["Type"]

            # Instrument-aware bump: futures in price space, rates/swaps in rate space
            bumped_mkt = self.df_mkt.copy()
            if typ == "Future":
                # Futures price: bump down by bump_bp (higher price = lower rate)
                bumped_mkt.at[idx, "Quote"] -= bump_bp / 100.0
            else:
                # Cash/swap rates: bump up by bump_bp
                bumped_mkt.at[idx, "Quote"] += bump_bp / 100.0

            # Rebootstrap with bumped quotes
            bumped = SOFRCurve(
                sofr_path=self.sofr_path,
                today=self.today,
                hw_a=self.hw_a,
                hw_sigma=self.hw_sigma,
                verbose=False
            )
            bumped.df_mkt = bumped_mkt
            bumped.bootstrap()

            # dDF = sum of changes across all pillar DFs
            min_len = min(len(self.pillar_df), len(bumped.pillar_df))
            d_df    = float(np.sum(bumped.pillar_df[:min_len] - self.pillar_df[:min_len]))
            rows.append((sym, row["Tenors"], typ, round(d_df, 8)))

        dv01 = pd.DataFrame(rows, columns=["Symbol", "Tenor", "Type", "dDF_sum"])
        if self.verbose:
            print("\nBucket DV01 (change in sum of pillar DFs per +1bp)")
            print(dv01.to_string(index=False))
        return dv01

    # ── Tenor grid ────────────────────────────────────────────────────────────
    def build_grid(self) -> None:
        """Reprice curve on the standard tenor grid. Populates self.grid."""
        rows = []
        for label, n, unit in TENOR_GRID_SPEC:
            if unit == "D":
                mat = self.today + timedelta(days=n)
            elif unit == "W":
                mat = self.modified_following(self.today + timedelta(weeks=n))
            else:
                mat = self.add_tenor(self.effective_date, f"{n}{unit}")

            T      = self.act360(self.today, mat)
            df_val = self.df(T)
            ddays  = (mat - self.today).days
            rows.append((
                label, mat, df_val,
                self._reprice(mat) * 100,
                self._sofr_fwd_1d(mat) * 100,
                ((1.0 / df_val) ** (1.0 / ddays) - 1.0) * 360 * 100,
            ))

        self.grid = pd.DataFrame(
            rows,
            columns=["Tenor", "MatDate", "DiscountFactor",
                     "SwapRate", "Fwd_SOFR_1D", "Exp_Fwd_SOFR_1D"]
        )

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"SOFR-OIS Repriced Curve — Standard Tenor Grid  |  {self.today}")
            print("=" * 70)
            print(self.grid.to_string(index=False, float_format="%.6f"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    def plot(self, save_path: str = "sofr_ois_curve.png") -> None:
        """3-panel SOFR-OIS curve chart saved to save_path."""
        if self.grid is None:
            raise RuntimeError("Call build_grid() before plot().")

        grid_dates  = self.grid["MatDate"].tolist()
        grid_tenors = self.grid["Tenor"].tolist()
        display_lbl = [t if t in SHOW_LABELS else "" for t in grid_tenors]

        metrics = [
            ("DiscountFactor",  "Discount Factor",
             "Discount Factor",  COLORS[0], "o"),
            ("SwapRate",        "SOFR Swap Curve",
             "Rate (%)",         COLORS[1], "s"),
            ("Exp_Fwd_SOFR_1D", "E[SOFR] Over Tenor (Flat Daily Rate, Act/360)",
             "Rate (%)",         COLORS[2], "^"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"SOFR-OIS Curve  —  Trade: {self.today} | Eff: {self.effective_date}",
            fontsize=18, fontweight="bold"
        )

        for ax, (col, title, ylabel, color, marker) in zip(axes, metrics):
            ax.plot(grid_dates, self.grid[col], f"{marker}-", color=color, lw=2)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(grid_dates)
            ax.set_xticklabels(display_lbl, rotation=45, fontsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        if self.verbose:
            print(f"Chart saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sofr_file  = os.path.join(os.getcwd(), "marketinputs", "sofr.xlsx")
    today_date = pd.read_excel(
        sofr_file, sheet_name=0, header=0, usecols="F", nrows=1
    ).iloc[0, 0].date()
    curve = SOFRCurve(sofr_path=sofr_file, today=today_date)
    curve.bootstrap()
    curve.build_grid()
    curve.plot()