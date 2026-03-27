# pricing_module.py
"""
Pricing primitives and tick-aware sensitivity helpers.

Provides:
 - black_scholes_vec, black_model_vec (vectorized)
 - unit pricing wrappers: unit_price_equity, unit_price_future, unit_price_option
 - apply_multiplier: convert unit price -> full contract cash (using futures_specs)
 - tick helpers and sensitivity functions: tick_round, linear_sens, option_sens, compute_sensitivities
 - volatility estimators (historical / ewma)
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Tuple, List, Any
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

_eps = 1e-12

# -------------------------
# Numerics helpers
# -------------------------
def _safe_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))

def _safe_div(numer, denom):
    denom_safe = np.where(np.abs(denom) < _eps, np.sign(denom) * _eps + _eps, denom)
    return numer / denom_safe

# -------------------------
# Black-Scholes (spot) - vectorized
# -------------------------
def black_scholes_vec(S, K, T, r, sigma, q=0.0, option_type="call"):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    shape = np.broadcast(S, K, T, r, sigma, q).shape
    S = np.broadcast_to(S, shape); K = np.broadcast_to(K, shape)
    T = np.broadcast_to(T, shape); r = np.broadcast_to(r, shape)
    sigma = np.broadcast_to(sigma, shape); q = np.broadcast_to(q, shape)

    price = np.zeros(shape); delta = np.zeros(shape); gamma = np.zeros(shape)
    vega = np.zeros(shape); theta = np.zeros(shape); rho = np.zeros(shape)

    zero_T = T <= 0
    zero_sigma = sigma <= 0
    mask = ~(zero_T | zero_sigma)

    if np.any(mask):
        Sm = S[mask]; Km = K[mask]; Tm = T[mask]; rm = r[mask]; sigmam = sigma[mask]; qm = q[mask]
        sqrtT = _safe_sqrt(Tm)
        d1 = (np.log(_safe_div(Sm, Km)) + (rm - qm + 0.5 * sigmam**2) * Tm) / (sigmam * sqrtT)
        d2 = d1 - sigmam * sqrtT
        Nd1 = norm.cdf(d1); Nd2 = norm.cdf(d2); n_d1 = norm.pdf(d1)

        if option_type == "call":
            price[mask] = np.exp(-qm * Tm) * Sm * Nd1 - np.exp(-rm * Tm) * Km * Nd2
            delta[mask] = np.exp(-qm * Tm) * Nd1
            rho[mask] = Km * Tm * np.exp(-rm * Tm) * Nd2
        else:
            price[mask] = np.exp(-rm * Tm) * Km * norm.cdf(-d2) - np.exp(-qm * Tm) * Sm * norm.cdf(-d1)
            delta[mask] = -np.exp(-qm * Tm) * norm.cdf(-d1)
            rho[mask] = -Km * Tm * np.exp(-rm * Tm) * norm.cdf(-d2)

        gamma[mask] = np.exp(-qm * Tm) * n_d1 / (Sm * sigmam * sqrtT)
        vega[mask] = Sm * np.exp(-qm * Tm) * n_d1 * sqrtT
        term1 = -Sm * np.exp(-qm * Tm) * n_d1 * sigmam / (2 * sqrtT)

        if option_type == "call":
            term2 = qm * Sm * np.exp(-qm * Tm) * norm.cdf(d1)
            term3 = -rm * Km * np.exp(-rm * Tm) * norm.cdf(d2)
            theta[mask] = term1 - term2 + term3
        else:
            term2 = qm * Sm * np.exp(-qm * Tm) * (-norm.cdf(-d1))
            term3 = rm * Km * np.exp(-rm * Tm) * norm.cdf(-d2)
            theta[mask] = term1 + term2 + term3

    # zero sigma but T>0 => intrinsic on forward
    mask_zero_sigma = (~zero_T) & zero_sigma
    if np.any(mask_zero_sigma):
        Sm = S[mask_zero_sigma]; Km = K[mask_zero_sigma]; Tm = T[mask_zero_sigma]; rm = r[mask_zero_sigma]; qm = q[mask_zero_sigma]
        F = Sm * np.exp((rm - qm) * Tm)
        if option_type == "call":
            price[mask_zero_sigma] = np.exp(-rm * Tm) * np.maximum(F - Km, 0.0)
            delta[mask_zero_sigma] = np.where(F > Km, np.exp(-qm * Tm), 0.0)
        else:
            price[mask_zero_sigma] = np.exp(-rm * Tm) * np.maximum(Km - F, 0.0)
            delta[mask_zero_sigma] = np.where(F < Km, -np.exp(-qm * Tm), 0.0)

    # T == 0 intrinsic
    if np.any(zero_T):
        Sm = S[zero_T]; Km = K[zero_T]
        if option_type == "call":
            price[zero_T] = np.maximum(Sm - Km, 0.0)
            delta[zero_T] = np.where(Sm > Km, 1.0, 0.0)
        else:
            price[zero_T] = np.maximum(Km - Sm, 0.0)
            delta[zero_T] = np.where(Sm < Km, -1.0, 0.0)

    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# -------------------------
# Black model (futures) - vectorized
# -------------------------
def black_model_vec(F, K, T, r, sigma, option_type="call"):
    F = np.asarray(F, dtype=float); K = np.asarray(K, dtype=float); T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float); sigma = np.asarray(sigma, dtype=float)

    shape = np.broadcast(F, K, T, r, sigma).shape
    F = np.broadcast_to(F, shape); K = np.broadcast_to(K, shape)
    T = np.broadcast_to(T, shape); r = np.broadcast_to(r, shape); sigma = np.broadcast_to(sigma, shape)

    price = np.zeros(shape); delta_f = np.zeros(shape); gamma_f = np.zeros(shape)
    vega = np.zeros(shape); theta = np.zeros(shape); rho = np.zeros(shape)

    zero_T = T <= 0; zero_sigma = sigma <= 0
    mask = ~(zero_T | zero_sigma)

    if np.any(mask):
        Fm = F[mask]; Km = K[mask]; Tm = T[mask]; rm = r[mask]; sigmam = sigma[mask]
        sqrtT = _safe_sqrt(Tm)
        d1 = (np.log(_safe_div(Fm, Km)) + 0.5 * sigmam**2 * Tm) / (sigmam * sqrtT)
        d2 = d1 - sigmam * sqrtT
        Nd1 = norm.cdf(d1); Nd2 = norm.cdf(d2); n_d1 = norm.pdf(d1)

        if option_type == "call":
            price[mask] = np.exp(-rm * Tm) * (Fm * Nd1 - Km * Nd2)
            delta_f[mask] = np.exp(-rm * Tm) * Nd1
        else:
            price[mask] = np.exp(-rm * Tm) * (Km * norm.cdf(-d2) - Fm * norm.cdf(-d1))
            delta_f[mask] = -np.exp(-rm * Tm) * norm.cdf(-d1)

        gamma_f[mask] = np.exp(-rm * Tm) * n_d1 / (Fm * sigmam * sqrtT)
        vega[mask] = Fm * np.exp(-rm * Tm) * n_d1 * sqrtT
        theta[mask] = -Fm * np.exp(-rm * Tm) * n_d1 * sigmam / (2 * sqrtT) + rm * price[mask]
        rho[mask] = -Tm * price[mask]

    # zero sigma but T>0: intrinsic on forward
    mask_zero_sigma = (~zero_T) & zero_sigma
    if np.any(mask_zero_sigma):
        Fm = F[mask_zero_sigma]; Km = K[mask_zero_sigma]; Tm = T[mask_zero_sigma]; rm = r[mask_zero_sigma]
        if option_type == "call":
            price[mask_zero_sigma] = np.exp(-rm * Tm) * np.maximum(Fm - Km, 0.0)
            delta_f[mask_zero_sigma] = np.where(Fm > Km, np.exp(-rm * Tm), 0.0)
        else:
            price[mask_zero_sigma] = np.exp(-rm * Tm) * np.maximum(Km - Fm, 0.0)
            delta_f[mask_zero_sigma] = np.where(Fm < Km, -np.exp(-rm * Tm), 0.0)

    # T == 0 intrinsic
    if np.any(zero_T):
        Fm = F[zero_T]; Km = K[zero_T]
        if option_type == "call":
            price[zero_T] = np.maximum(Fm - Km, 0.0)
            delta_f[zero_T] = np.where(Fm > Km, 1.0, 0.0)
        else:
            price[zero_T] = np.maximum(Km - Fm, 0.0)
            delta_f[zero_T] = np.where(Fm < Km, -1.0, 0.0)

    return {"price": price, "delta_f": delta_f, "gamma_f": gamma_f, "vega": vega, "theta": theta, "rho": rho}


# -------------------------
# High-level unit pricers
# -------------------------
def unit_price_equity(S: float) -> Dict[str, float]:
    """Unit price for equity (per share). Returns dict(price, delta=1)."""
    return {"price": float(S), "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

def unit_price_future(F: float) -> Dict[str, float]:
    """Unit price for a future contract (price expressed in futures price units)."""
    # For mark-to-market we typically use P&L (difference). This returns unit price as the futures quote.
    return {"price": float(F), "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

def unit_price_option(model="auto", S=None, F=None, K=None, T=None, r=0.0, sigma=0.2, q=0.0,
                      option_type="call", underlying_is_currency=False, r_dom=None, r_for=None):
    """
    Return unit price and unit greeks (per contract unit) WITHOUT applying multiplier/notional.
    model: "auto"/"bs"/"black". If F provided -> Black; if S provided -> BS.
    """
    use_black = (model == "black") or (model == "auto" and F is not None)
    if use_black:
        if F is None:
            # attempt to compute forward from spot/currency rates
            if underlying_is_currency and (r_dom is not None and r_for is not None) and S is not None and T is not None:
                F = S * np.exp((r_dom - r_for) * T)
            elif (r is not None) and (S is not None) and (T is not None):
                F = S * np.exp(r * T)
            else:
                raise ValueError("Black model requires F or S + rates to compute F.")
        out = black_model_vec(F=F, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        # convert delta wrt futures to delta wrt spot (if currency underlying)
        if underlying_is_currency and (r_dom is not None and r_for is not None) and T is not None:
            conv = np.exp((r_dom - r_for) * T)
            delta_spot = out["delta_f"] * conv
        else:
            delta_spot = out["delta_f"]
        return {"price": float(out["price"]), "delta": float(delta_spot), "gamma": float(out["gamma_f"]),
                "vega": float(out["vega"]), "theta": float(out["theta"]), "rho": float(out["rho"])}
    else:
        out = black_scholes_vec(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
        return {"price": float(out["price"]), "delta": float(out["delta"]), "gamma": float(out["gamma"]),
                "vega": float(out["vega"]), "theta": float(out["theta"]), "rho": float(out["rho"])}

# -------------------------
# Multiply unit -> contract cash
# -------------------------
def apply_multiplier(unit_price: float, symbol: str, futures_specs: Dict[str, dict] = None, default_multiplier: float = 1.0):
    """
    Convert unit price to contract-level cash using futures_specs. If not present use default_multiplier.
    Example: CL unit price is quoted per barrel => multiplier 1000 -> full cash.
    """
    futures_specs = futures_specs or {}
    mult = futures_specs.get(symbol, {}).get("multiplier", default_multiplier)
    return float(unit_price) * float(mult)

# -------------------------
# Tick helpers & sensitivities
# -------------------------
def tick_round(value: float, tick: float) -> float:
    """Round value to nearest multiple of tick (half-up)."""
    if tick is None or tick <= 0:
        return float(value)
    v = Decimal(str(value)); t = Decimal(str(tick))
    n = (v / t); n_q = n.to_integral_value(rounding=ROUND_HALF_UP)
    return float(n_q * t)

def linear_sens(S: float, pct_move: float = 0.01, tick_size: float = 0.01) -> float:
    """Tick-adjusted underlying absolute move for given percent move (returns price units)."""
    raw = pct_move * S
    mv = tick_round(raw, tick_size)
    if mv == 0.0 and abs(raw) > 0:
        mv = tick_size if raw > 0 else -tick_size
    return mv

def option_sens(pos: dict, U: float, r: float, sigma: float, pct_move: float = 0.01, tick_size: float = 0.01,
                futures_specs: Dict[str, dict] = None):
    """
    Per-contract option price change for pct_move in underlying, tick-adjusted.
    Returns dict with linear/convex components and unit greeks.
    """
    K = float(pos["strike"])
    if "expiry_days" in pos:
        expiry_days = float(pos["expiry_days"])
    else:
        if "expiry" in pos:
            ed = datetime.datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
            expiry_days = max((ed - datetime.date.today()).days, 0)
        else:
            expiry_days = pos.get("T_days", 30)
    T = max(expiry_days / 365.0, 1e-6)

    cp = pos["option_type"].lower()
    underlying_type = pos.get("underlying_type", "equity")
    model = "black" if underlying_type == "future" else "bs"

    if model == "black":
        unit = unit_price_option(model="black", F=U, K=K, T=T, r=r, sigma=sigma, option_type=cp)
    else:
        unit = unit_price_option(model="bs", S=U, K=K, T=T, r=r, sigma=sigma, q=pos.get("q", 0.0), option_type=cp)

    raw = pct_move * U
    move_size = tick_round(raw, tick_size)
    if move_size == 0.0 and abs(raw) > 0:
        move_size = tick_size if raw > 0 else -tick_size

    chg_lin = unit["delta"] * move_size
    chg_conv = 0.5 * unit["gamma"] * (move_size ** 2)
    price_chg = chg_lin + chg_conv

    return {"price_chg": price_chg, "chg_lin": chg_lin, "chg_conv": chg_conv, "unit_delta": unit["delta"], "unit_gamma": unit["gamma"], "unit": unit}

def compute_sensitivities(pos: dict,
                          prices: dict,
                          vols: dict,
                          futures_specs: dict = None,
                          pct_move: float = 0.01,
                          r: float = 0.04):
    """
    Compute tick-aware sensitivities for a position.
    Returns dict:
      { "base": cash sensitivity for pct_move,
        "vol_1d": cash sensitivity for 1-day sigma move (or None),
        "details": {...} }
    """
    futures_specs = futures_specs or {}
    sym = pos.get("symbol"); typ = pos.get("type"); qty = float(pos.get("quantity", 0))

    # infer tick and contract multiplier
    if typ == "future" or (typ == "option" and pos.get("underlying_type") == "future"):
        fut_sym = pos.get("underlying_symbol", sym)
        spec = futures_specs.get(fut_sym, {})
        tick_size = float(spec.get("tick_size", 0.01)); mult = float(spec.get("multiplier", 1.0))
    else:
        tick_size = 0.01; mult = 100.0 if typ == "option" else 1.0

    und_sym = pos.get("underlying_symbol", sym)
    U = prices.get(und_sym, prices.get(sym))
    if U is None:
        return {"base": 0.0, "vol_1d": None, "details": {"error": "missing underlying price"}}

    details = {}

    if typ == "option":
        sigma = vols.get(sym, vols.get(und_sym, pos.get("vol", 0.20)))
        opt = option_sens(pos, U, r, sigma, pct_move, tick_size, futures_specs)
        base_cash = opt["price_chg"] * qty * mult
        details["pct"] = {"raw_move": pct_move * U, "tick_move": tick_round(pct_move * U, tick_size),
                          "per_contract_price_chg": opt["price_chg"], "chg_lin": opt["chg_lin"],
                          "chg_conv": opt["chg_conv"], "unit_delta": opt["unit_delta"], "unit_gamma": opt["unit_gamma"],
                          "qty": qty, "multiplier": mult}
    else:
        move_tick = linear_sens(U, pct_move, tick_size)
        base_cash = move_tick * qty * mult
        details["pct"] = {"raw_move": pct_move * U, "tick_move": move_tick, "qty": qty, "multiplier": mult}

    # vol-normalized (1-day sigma)
    vol_key = sym if typ != "option" else pos.get("underlying_symbol", sym)
    sigma = vols.get(sym, vols.get(vol_key, None))
    if sigma is not None and sigma > 0:
        raw_vol_move = sigma * U / (252 ** 0.5)
        vol_tick = tick_round(raw_vol_move, tick_size)
        if vol_tick == 0.0 and raw_vol_move != 0.0:
            vol_tick = tick_size if raw_vol_move > 0 else -tick_size

        if typ == "option":
            opt_vol = option_sens(pos, U, r, sigma, pct_move=(vol_tick / U if U else 0.0), tick_size=tick_size, futures_specs=futures_specs)
            vol_cash = opt_vol["price_chg"] * qty * mult
            details["vol_1d"] = {"raw_move": raw_vol_move, "tick_move": vol_tick, "per_contract_price_chg": opt_vol["price_chg"], "qty": qty, "multiplier": mult}
        else:
            vol_cash = vol_tick * qty * mult
            details["vol_1d"] = {"raw_move": raw_vol_move, "tick_move": vol_tick, "qty": qty, "multiplier": mult}
    else:
        vol_cash = None
        details["vol_1d"] = {"note": "no vol available"}

    return {"base": base_cash, "vol_1d": vol_cash, "details": details}


# -------------------------
# Vol estimators
# -------------------------
def historical_volatility(prices, window=252) -> pd.Series:
    series = pd.Series(prices).dropna()
    if len(series) < 2:
        return pd.Series(dtype=float)
    logret = np.log(series).diff().dropna()
    return logret.rolling(window=window).std() * np.sqrt(252)

def ewma_volatility(prices, span=63) -> pd.Series:
    series = pd.Series(prices).dropna()
    if len(series) < 2:
        return pd.Series(dtype=float)
    logret = np.log(series).diff().dropna()
    var_ewma = logret.ewm(span=span).var()
    return np.sqrt(var_ewma) * np.sqrt(252)
# -------------------------
# End of pricing_module.py
# -------------------------