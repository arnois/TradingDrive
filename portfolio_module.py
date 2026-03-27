# portfolio_module.py
"""
Pricing primitives delegated to pricing_module, and PortfolioManager with
tick-aware sensitivities. Enhanced:
 - mark_to_market returns per-position unit vs contract metrics
 - mark_to_market_views returns summaries by asset class and equities vs derivatives
 - update_positions to apply minimal changes (expiry_days, trade_price, ...)
 - global short-rate support via set_global_short_rate/get_global_short_rate
"""

from typing import Dict, Tuple, List, Any
import datetime
import numpy as np

# import pricing helpers (moved to pricing_module)
from pricing_module import (
    unit_price_option,
    unit_price_equity,
    unit_price_future,
    apply_multiplier,
    tick_round,
    compute_sensitivities,
    historical_volatility,
    ewma_volatility,
)

# -------------------------
# Global short-rate (can be updated via set_global_short_rate)
# -------------------------
GLOBAL_SHORT_RATE = 0.04


def set_global_short_rate(rate: float):
    global GLOBAL_SHORT_RATE
    GLOBAL_SHORT_RATE = float(rate)


def get_global_short_rate() -> float:
    return GLOBAL_SHORT_RATE


# -------------------------
# Helper: option key formatter (unchanged)
# -------------------------
def _opt_key_from_position(pos: dict) -> str:
    try:
        expiry = datetime.datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
    except Exception:
        sysdate = datetime.datetime.today()
        expiry = sysdate + datetime.timedelta(days=int(pos['expiry_days']))
        expiry = expiry.date()
    mon = expiry.strftime("%b").upper(); yy = str(expiry.year)[-2:]
    cp = "C" if pos["option_type"].lower().startswith("c") else "P"
    strike_txt = f"{pos['strike']}".rstrip("0").rstrip(".") if isinstance(pos["strike"], float) else str(pos["strike"])
    return f"{pos.get('underlying_symbol', pos['symbol'])}_{mon}{yy}_{strike_txt}{cp}"


# -------------------------
# PortfolioManager
# -------------------------
class PortfolioManager:
    """
    positions: list of dicts, each dict must include keys:
      - type: "equity" | "future" | "option"
      - symbol: root symbol
      - quantity: signed (positive long, negative short)
    Options must include: strike, expiry (YYYY-MM-DD) or expiry_days, option_type ("call"/"put"),
    underlying_type ("equity"/"future") and optionally underlying_symbol.
    futures_specs: dict keyed by future symbol with {"multiplier":float, "tick_size":float, "asset_class":...}
    """
    def __init__(self, positions: List[dict], futures_specs: Dict[str, dict] = None):
        self.positions = positions or []
        self.futures_specs = futures_specs or {}

    def _fut_mult(self, sym: str) -> float:
        return float(self.futures_specs.get(sym, {}).get("multiplier", 1.0))

    def _asset_class_for_symbol(self, sym: str) -> str:
        return self.futures_specs.get(sym, {}).get("asset_class", "Unknown")

    # -------------------------
    # Update positions in-place (minimal changes)
    # updates: list of dicts with at least {"symbol":..., optional "type":...} and fields to change
    # Returns (n_updated, details_list)
    # -------------------------
    def update_positions(self, updates: List[dict]) -> Tuple[int, List[dict]]:
        updated = 0
        details = []
        for up in updates:
            sym = up.get("symbol")
            typ = up.get("type", None)
            matched = False
            for pos in self.positions:
                if pos.get("symbol") != sym:
                    continue
                if typ and pos.get("type") != typ:
                    continue
                # apply updates (only top-level keys)
                changed_fields = {}
                for k, v in up.items():
                    if k in ("symbol", "type"):
                        continue
                    # only change if value differs (or if missing)
                    old = pos.get(k, None)
                    if old != v:
                        pos[k] = v
                        changed_fields[k] = {"old": old, "new": v}
                if changed_fields:
                    updated += 1
                    details.append({"symbol": sym, "type": pos.get("type"), "changes": changed_fields})
                else:
                    details.append({"symbol": sym, "type": pos.get("type"), "changes": None})
                matched = True
                # do not apply same update to multiple positions with same symbol unless user expects that
                break
            if not matched:
                details.append({"symbol": sym, "status": "not_found"})
        return updated, details

    # -------------------------
    # Bumped MTM: similar semantics to prior implementation but returns contract-level cash
    # bump is price units added to underlying
    # -------------------------
    def bumped_mtm(self, pos: dict, prices: dict, vols: dict, bump: float = 0.0, r: float = None, today: datetime.date = None) -> float:
        today = today or datetime.date.today()
        r = get_global_short_rate() if r is None else r

        typ = pos.get("type"); sym = pos.get("symbol"); qty = float(pos.get("quantity", 0))
        und = pos.get("underlying_symbol", sym)
        U_base = prices.get(und, prices.get(sym))
        if U_base is None:
            return 0.0
        U = U_base + bump

        if typ == "equity":
            return qty * U

        if typ == "future":
            mult = float(self.futures_specs.get(sym, {}).get("multiplier", 1.0))
            entry_unit = pos.get("trade_price", U_base)
            # contract-level MTM P&L
            return (U - entry_unit) * qty * mult

        if typ == "option":
            K = float(pos["strike"])
            if "expiry" in pos:
                ed = datetime.datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
                T = max((ed - today).days / 365.0, 1e-6)
            else:
                T = max(pos.get("expiry_days", 30) / 365.0, 1e-6)
            sigma = vols.get(pos.get("symbol"), vols.get(und, pos.get("vol", 0.2)))
            if pos.get("underlying_type") == "future":
                unit = unit_price_option(model="black", F=U, K=K, T=T, r=r, sigma=sigma, option_type=pos.get("option_type","call"))
                mult = float(self.futures_specs.get(und, {}).get("multiplier", 1.0))
                entry_unit = pos.get("trade_price", 0.0)
                return (unit["price"] - entry_unit) * qty * mult
            else:
                unit = unit_price_option(model="bs", S=U, K=K, T=T, r=r, sigma=sigma, option_type=pos.get("option_type","call"), q=pos.get("q", 0.0))
                entry_unit = pos.get("trade_price", 0.0)
                return (unit["price"] - entry_unit) * qty * 100.0

        return 0.0

    # -------------------------
    # mark_to_market (enhanced row info)
    # Returns (total_value, breakdown_list) like before but breakdown rows include:
    #  - unit_price, unit_entry_price, unit_pnl, contract_multiplier, market_value, mtm_pnl, greeks, exposures, sensitivities, validation (if any)
    # total_value is sum of contract-level market_values (keeps previous behavior where futures MV was MTM P&L only?).
    # We'll define total_value as sum of contract-level market_value (unit_price*qty*mult) for consistency.
    # If you prefer MTM-sum only, pass r and compute bumped values separately.
    # -------------------------
    def mark_to_market(self,
                       prices: Dict[str, float],
                       vols: Dict[str, float] = None,
                       unit_prices_options: Dict[str, float] = None,
                       r: float = None,
                       today: datetime.date = None,
                       pct_move: float = 0.01) -> Tuple[float, List[Dict[str, Any]]]:
        vols = vols or {}
        unit_prices_options = unit_prices_options or {}
        today = today or datetime.date.today()
        r = get_global_short_rate() if r is None else r

        total_value = 0.0
        breakdown = []

        for pos in self.positions:
            if not pos.get("open", True):
                continue

            ptype = pos.get("type"); sym = pos.get("symbol"); qty = float(pos.get("quantity", 0))
            tpx = pos.get("trade_price", None)

            # defaults for reporting
            unit_price = None
            unit_entry = tpx if tpx is not None else None
            unit_pnl = None
            contract_multiplier = 1.0
            market_value = 0.0
            mtm_pnl = 0.0

            greeks = {"delta": None, "gamma": None, "vega": None, "theta": None, "rho": None}
            expos = {"delta_units": None, "delta_notional": None, "gross_notional": None}
            notes = {}

            # equities
            if ptype == "equity":
                S = prices.get(sym)
                if S is None:
                    # skip if no price
                    continue
                unit_price = float(S)
                unit_entry = float(unit_entry) if unit_entry is not None else None
                contract_multiplier = 1.0
                market_value = qty * unit_price
                unit_pnl = (unit_price - unit_entry) * qty if unit_entry is not None else None
                mtm_pnl = unit_pnl  # same because multiplier=1
                greeks["delta"] = qty
                expos["delta_units"] = qty
                expos["delta_notional"] = qty * unit_price
                expos["gross_notional"] = abs(qty * unit_price)

            # futures
            elif ptype == "future":
                F_mark = prices.get(sym)
                if F_mark is None:
                    continue
                unit_price = float(F_mark)
                unit_entry = float(unit_entry) if unit_entry is not None else unit_price
                mult = self._fut_mult(sym)
                contract_multiplier = float(mult)
                unit_pnl = (unit_price - unit_entry) * qty
                market_value = unit_pnl * contract_multiplier # market_value as contract-level pnl
                mtm_pnl = (unit_price - unit_entry) * qty * contract_multiplier
                expos["delta_units"] = qty * contract_multiplier
                expos["delta_notional"] = expos["delta_units"] * unit_price
                expos["gross_notional"] = abs(unit_price * qty * contract_multiplier)
                greeks["delta"] = expos["delta_units"]

            # options
            elif ptype == "option":
                K = float(pos["strike"])
                if "expiry" in pos:
                    ed = datetime.datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
                    T = max((ed - (today or datetime.date.today())).days / 365.0, 1e-6)
                else:
                    T = max(pos.get("expiry_days", 30) / 365.0, 1e-6)
                cp = pos.get("option_type", "call").lower()
                # determine option unit price via model (unit-level)
                vol = vols.get(sym, pos.get("vol", 0.20))

                unit_key = None
                try:
                    unit_key = _opt_key_from_position(pos)
                except Exception:
                    print("No Unit Key Provided Nor Found")
                    unit_key = None
                unit_px_provided = unit_prices_options.get(unit_key) if unit_key else None

                # futures options
                if pos.get("underlying_type") == "future":
                    und_sym = pos.get("underlying_symbol", sym)
                    mult = self._fut_mult(und_sym)
                    contract_multiplier = float(mult)
                    F_mark = prices.get(und_sym, prices.get(sym))
                    if F_mark is None:
                        continue
                    pr_unit = unit_price_option(model="black",
                                                F=F_mark,
                                                K=K,
                                                T=T,
                                                r=r,
                                                sigma=vol,
                                                option_type=cp)
                    #unit_price = float(pr_unit["price"])
                    unit_pr_tick_size = float(self.futures_specs.get(und_sym[:-3]).get('tick_size'))
                    unit_price = tick_round(float(pr_unit["price"]),
                                            unit_pr_tick_size)
                    # validation vs provided unit px if present
                    if unit_px_provided is not None:
                        diff_unit = tick_round(unit_price - unit_px_provided, unit_pr_tick_size)
                        pct_diff = 100*np.round(diff_unit / unit_px_provided,4) \
                            if unit_px_provided != 0 else float("inf")
                        notes.update({"unit_price_provided": unit_px_provided,
                                      "model_unit_price": unit_price,
                                      "unit_vs_model_diff": diff_unit,
                                      "unit_vs_model_pct": pct_diff})
                    unit_entry = float(unit_entry) if unit_entry is not None else None
                    # market value as contract notional (unit_price * qty * mult)
                    market_value = unit_price * qty * contract_multiplier
                    unit_pnl = (unit_price - unit_entry) * qty if unit_entry is not None else None
                    mtm_pnl = (unit_price - unit_entry) * qty * contract_multiplier if unit_entry is not None else None
                    greeks = {k: pr_unit.get(k) * qty * contract_multiplier for k in ("delta", "gamma", "vega", "theta", "rho")}
                    expos["delta_units"] = pr_unit["delta"] * qty * contract_multiplier
                    expos["delta_notional"] = expos["delta_units"] * F_mark
                    expos["gross_notional"] = abs(F_mark * qty * contract_multiplier)

                # equity options
                else:
                    S = prices.get(pos.get("underlying_symbol", sym), prices.get(sym))
                    if S is None:
                        continue
                    pr_unit = unit_price_option(model="bs",
                                                S=S,
                                                K=K,
                                                T=T,
                                                r=pos.get("rate", r),
                                                sigma=vol,
                                                q=pos.get("q", 0.0),
                                                option_type=cp)
                    #unit_price = float(pr_unit["price"])
                    unit_pr_tick_size = float(0.01)
                    unit_price = tick_round(float(pr_unit["price"]),
                                            unit_pr_tick_size)
                    if unit_px_provided is not None:
                        diff_unit = tick_round(unit_price - unit_px_provided, unit_pr_tick_size)
                        pct_diff = 100 * np.round(diff_unit / unit_px_provided, 4) \
                            if unit_px_provided != 0 else float("inf")
                        notes.update({"unit_price_provided": unit_px_provided,
                                      "model_unit_price": unit_price,
                                      "unit_vs_model_diff": diff_unit,
                                      "unit_vs_model_pct": pct_diff})
                    unit_entry = float(unit_entry) if unit_entry is not None else None
                    contract_multiplier = 100.0
                    market_value = unit_price * qty * contract_multiplier
                    unit_pnl = (unit_price - unit_entry) * qty if unit_entry is not None else None
                    mtm_pnl = (unit_price - unit_entry) * qty * contract_multiplier if unit_entry is not None else None
                    greeks = {k: pr_unit.get(k) * qty * contract_multiplier for k in ("delta", "gamma", "vega", "theta", "rho")}
                    expos["delta_units"] = pr_unit["delta"] * qty * contract_multiplier
                    expos["delta_notional"] = expos["delta_units"] * S
                    expos["gross_notional"] = abs(S * qty * contract_multiplier)

            else:
                # unknown type
                continue

            # sensitivities (tick-aware)
            sens = compute_sensitivities(pos, prices, vols, self.futures_specs, pct_move, r)

            # aggregate total_value: using contract-level market_value (not MTM-only)
            total_value += market_value

            row = {
                **pos,
                "unit_price": unit_price,
                "unit_entry_price": unit_entry,
                "unit_pnl": unit_pnl,
                "contract_multiplier": contract_multiplier,
                "market_value": market_value,
                "mtm_pnl": mtm_pnl,
                "greeks": greeks,
                "exposures": expos,
                "sensitivities": sens
            }
            if notes:
                row["validation"] = notes
            breakdown.append(row)

        return total_value, breakdown

    # -------------------------
    # Convenience: produce summarized views
    # Returns (total_value, breakdown, views) where views is dict with:
    #   - by_asset_class: {asset_class: {market_value, gross_notional, count}}
    #   - equities_vs_derivatives: {"EQUITY": {...}, "DERIVATIVE": {...}}
    # -------------------------
    def mark_to_market_views(self,
                             prices: Dict[str, float],
                             vols: Dict[str, float] = None,
                             unit_prices_options: Dict[str, float] = None,
                             r: float = None,
                             today: datetime.date = None,
                             pct_move: float = 0.01) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
        total_value, breakdown = self.mark_to_market(prices=prices,
                                                     vols=vols,
                                                     unit_prices_options=unit_prices_options,
                                                     r=r,
                                                     today=today,
                                                     pct_move=pct_move)

        # summarize by asset_class (from futures_specs or "EQUITY"/"Unknown")
        by_asset = {}
        eq_vs_der = {"EQUITY": {"market_value": 0.0, "gross_notional": 0.0, "count": 0},
                     "DERIVATIVE": {"market_value": 0.0, "gross_notional": 0.0, "count": 0}}

        for row in breakdown:
            ptype = row.get("type")
            sym = row.get("symbol")
            mv = row.get("market_value", 0.0) or 0.0
            gn = row.get("exposures", {}).get("gross_notional", 0.0) or 0.0
            # determine asset_class:
            isEquOpt = ('underlying_type' in row) and (row.get('underlying_type') == 'equity')
            isEqu = (ptype == "equity") or isEquOpt
            if isEqu:
                asset = "Equity"
            else:
                asset = self._asset_class_for_symbol(sym[:-3]) or "DerivUnknown"
            # aggregate
            ba = by_asset.setdefault(asset, {"market_value": 0.0, "gross_notional": 0.0, "count": 0})
            ba["market_value"] += mv
            ba["gross_notional"] += gn
            ba["count"] += 1

            # eq vs derivatives
            if ptype == "equity":
                bucket = "EQUITY"
            else:
                bucket = "DERIVATIVE"
            eq_vs_der[bucket]["market_value"] += mv
            eq_vs_der[bucket]["gross_notional"] += gn
            eq_vs_der[bucket]["count"] += 1

        views = {"by_asset_class": by_asset, "equities_vs_derivatives": eq_vs_der}

        return total_value, breakdown, views

    # -------------------------
    # Thin wrappers to expose vol estimators
    # -------------------------
    def historical_volatility_series(self, prices, window=252):
        return historical_volatility(prices, window=window)

    def ewma_volatility_series(self, prices, span=63):
        return ewma_volatility(prices, span=span)

# -------------------------
# End of module
# -------------------------
