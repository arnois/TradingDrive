#%% USERMANUALLY
# F,K,T,r,sigma,cp = 8.32, 8, 64/360, 0.03625, 0.6070,'call'
# unit_price_option(model='auto',F=F,K=K,T=T,r=r,sigma=sigma,option_type=cp)

#%% LIBS & CONFIGS
import os
import glob
import json
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
# Import the PortfolioManager (patched) and new pricing primitives
from portfolio_module import PortfolioManager
from pricing_module import unit_price_option, apply_multiplier, tick_round
from matplotlib import pyplot as plt

# ENV PATHS
os.environ["DATA_ROOT"] = r"C:\dev\data"
os.environ["DATA_EQU"] = os.path.join(os.environ["DATA_ROOT"], 'equ')
os.environ["DATA_FUT"] = os.path.join(os.environ["DATA_ROOT"], 'futures')

# GLOBS
FILE_EQU = "equities_data_msci.csv"
PATTERN_FILE_FUT = "futures_data_msci_*d*.csv"
FILE_FUT_SPECS = "futures_specs.txt"
fut_specs_path = os.path.join(os.environ["DATA_FUT"], FILE_FUT_SPECS)
if not Path(fut_specs_path).exists():
    print(f"[WARN] Futures specs file not found at: {fut_specs_path}. "+\
          "Using empty registry.")
    FUTURES_SPECS = {}
else:
    with open(fut_specs_path, "r") as f:
        FUTURES_SPECS = json.load(f)

#%% UDF
def get_futures_specs(symbol, specs_registry=FUTURES_SPECS,
                      default_multiplier=1, default_tick=1.0):
    if symbol in specs_registry:
        return specs_registry[symbol]
    return {
        "multiplier": default_multiplier,
        "tick_size": default_tick,
        "tick_value": default_multiplier * default_tick,
        "asset_class": "Unknown"
    }

def load_equities(path=None):
    path = path or os.path.join(os.environ["DATA_EQU"], FILE_EQU)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Equities file not found at {path}")
    df = pd.read_csv(path, parse_dates=["TRADING_DATE"], dayfirst=False)
    # Remove exchange suffix (e.g. 'AAPL.O' -> 'AAPL')
    if "RIC" in df.columns:
        df["RIC"] = df["RIC"].astype(str).str.split(".").str[0]
    df = df.set_index("TRADING_DATE").sort_index()
    return df

def load_futures(path_pattern=None):
    path_pattern = path_pattern or os.path.join(os.environ["DATA_FUT"],
                                                PATTERN_FILE_FUT)
    files = glob.glob(path_pattern)
    if not files:
        print(f"[WARN] No futures files found with pattern: {path_pattern}")
        return pd.DataFrame()  # empty df
    dfs = []
    for fpath in sorted(files):
        dfi = pd.read_csv(fpath, parse_dates=["OBS_DATE"], dayfirst=False)
        dfs.append(dfi)
    fut = pd.concat(dfs, ignore_index=True)
    fut = fut.set_index("OBS_DATE").sort_index()
    return fut

def select_futures_contracts(df, default_rules, custom_rules=None):
    """
    Select matching futures contracts (returns DataFrame).
    If df empty, returns empty df.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    rules = default_rules.copy()
    if custom_rules:
        rules.update(custom_rules)
    selected = []
    for sym, rule in rules.items():
        # Filter: PX_IDENT might contain root or exact ident depending on file;
        # try exact and prefix
        mask = (df.get("PX_IDENT") == sym) & (df.get("MATURITY") == rule)
        if not mask.any():
            # try prefix match in case file uses full contract names
            mask = df.get("PX_IDENT", "").astype(str).str.startswith(sym) & \
                   (df.get("MATURITY") == rule)
        if mask.any():
            selected.append(df[mask])
    if not selected:
        return pd.DataFrame()
    sel_fut = pd.concat(selected, ignore_index=False)
    return sel_fut

def merge_data(eq_df, fut_df):
    """
    Merge equities and futures into tidy xarray.Dataset keyed by (symbol, date).
    Returns xr.Dataset with variables price and volume and attribute asset_type
    per symbol.
    """
    pieces = []
    # equities
    if eq_df is not None and not eq_df.empty:
        eq = eq_df.copy()
        # Ensure required columns exist
        for c in ("RIC", "LAST", "UNSCALED_VOLUME"):
            if c not in eq.columns:
                raise KeyError(f"Equities frame missing column {c}")
        eq2 = eq[["RIC", "LAST", "UNSCALED_VOLUME"]].reset_index().rename(
            columns={"TRADING_DATE": "DATE", "RIC": "SYMBOL",
                     "LAST": "PRICE", "UNSCALED_VOLUME": "VOLUME"}
        )
        eq2["ASSET_TYPE"] = "EQUITY"
        pieces.append(eq2)
    # futures
    if fut_df is not None and not fut_df.empty:
        fut = fut_df.copy()
        # expect columns PX_IDENT and PRICE
        price_col = "PRICE" if "PRICE" in fut.columns else \
            ("LAST" if "LAST" in fut.columns else None)
        if price_col is None:
            raise KeyError("Futures dataframe missing PRICE/LAST column")
        fut2 = fut.reset_index()\
                .rename(columns={"OBS_DATE": "DATE",
                                 "PX_IDENT": "SYMBOL",
                                 price_col: "PRICE"})
        fut2["VOLUME"] = np.nan
        fut2["ASSET_TYPE"] = "FUTURE"
        pieces.append(fut2[["DATE", "SYMBOL", "PRICE", "VOLUME", "ASSET_TYPE"]])

    if not pieces:
        # return empty dataset
        return xr.Dataset()

    merged = pd.concat(pieces, ignore_index=True)
    # pivot to time x symbol, keep price & volume
    merged["DATE"] = pd.to_datetime(merged["DATE"])
    merged = merged.sort_values(["SYMBOL", "DATE"])
    # create xarray Dataset
    ds = merged.set_index(["SYMBOL", "DATE"]).to_xarray()
    # ds will have variables PRICE and VOLUME indexed by SYMBOL and DATE
    return ds

def first_nonnull(s):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan

#%% DATABASE LOAD
equ_df = load_equities()
fut_df = load_futures()
# futures maturity contract
default_rules = dict(zip(fut_df.PX_IDENT.unique(),
                         ["3M"] * len(fut_df.PX_IDENT.unique())))
default_rules['Y10Y'] = '1M'
fut_df_sel = select_futures_contracts(fut_df, default_rules)

#%% MOCK PFOLIO
symbols = ['TY', 'ES', 'CL', 'URO', 'GC']
window = int(21*6)
date_col = 'OBS_DATE'
# Futures datatable
df_sym = fut_df_sel[fut_df_sel['PX_IDENT'].isin(symbols)]
price_wide = (
    df_sym
    .reset_index()
    .pivot_table(
        index=df_sym.index.name or 'index',
        columns='PX_IDENT',
        values='PRICE',
        aggfunc=lambda s: first_nonnull(pd.Series(s))
    )
)
# Returns data
ret = price_wide.apply(np.log).diff()
ret = ret.loc[~ret.index.duplicated()]
ret = ret.sort_index()
ret = ret.dropna()

# Rolling COV/CORR
rolling_cov = {}
rolling_corr = {}
for i in range(window - 1, len(ret)):
    current_date = ret.index[i]
    window_df = ret.iloc[i - window + 1 : i + 1]  # shape (window, n_symbols)
    # you may choose to drop columns with too many NaNs in the window:
    # window_df = window_df.dropna(axis=1, thresh=int(window*0.5))
    cov_mat = window_df.cov()
    corr_mat = window_df.corr()
    rolling_cov[current_date] = cov_mat
    rolling_corr[current_date] = corr_mat


# CORR
first_date = list(rolling_cov.keys())[0]
last_date = list(rolling_cov.keys())[-1]
print("Corr matrix at", first_date)
print(rolling_corr[first_date])

# MIN VAR PFL
cov = rolling_cov[first_date]
ones = np.ones(len(cov))
inv_cov = np.linalg.inv(cov.values)
w = inv_cov @ ones
w /= ones @ inv_cov @ ones  # normalize to sum to 1
weights = pd.Series(w, index=cov.columns, name='GMV_weight')

# PFL RISK
pfl_ret = ret.iloc[0:window] @ weights
std_pfl = pfl_ret.std()

# PFL RISK AS XSR
S = pd.Series(np.diag(rolling_cov[first_date]), index=rolling_cov[first_date].columns).apply(np.sqrt)
R = pd.concat([pfl_ret,ret.iloc[0:window]], axis=1).corr()[0].drop(0)
XS = weights*S # Vol of return contributions
SR = S*R # MCR
XSR = weights*S*R
pfl_risk = XSR.sum()

# FUTURES PRICE-RET EXPOSURE
tkr_rename = {'TY': 'ZN', 'URO':'M6E'}
# 1-prcnt move exposure
px_chg = 0.01*price_wide.loc[first_date]
px_chg = px_chg.rename(tkr_rename)
px_chg_ticks = pd.Series(
    [tick_round(v,FUTURES_SPECS.get(tkr)['tick_size']) for tkr,v in px_chg.items()],
    index = px_chg.index
)
px_chg_sens = pd.Series(
    [v*FUTURES_SPECS.get(tkr)['multiplier'] for tkr,v in px_chg_ticks.items()],
    index = px_chg_ticks.index
)

# 1-std move exposure
px_chg_1S = (S*price_wide.loc[first_date]).rename(tkr_rename)
px_chg_1S_ticks = pd.Series(
    [tick_round(v,FUTURES_SPECS.get(tkr)['tick_size']) for tkr,v in px_chg_1S.items()],
    index = px_chg_1S.index
)
px_chg_1S_sens = pd.Series( # 1-std PnL per asset
    [v*FUTURES_SPECS.get(tkr)['multiplier'] for tkr,v in px_chg_1S_ticks.items()],
    index = px_chg_1S_ticks.index
)

# PFLIO :: GMV Weights
init_amt = 1e7
init_px = pd.Series(
    [tick_round(p,FUTURES_SPECS.get(tk)['tick_size']) for tk,p in price_wide.
                                                                    loc[first_date].
                                                                    rename(tkr_rename).items()],
    index = price_wide.loc[first_date].rename(tkr_rename).index)
init_px_amt = pd.Series(
    [p*FUTURES_SPECS.get(tk)['multiplier'] for tk,p in init_px.items()],
    index = init_px.index
)
w_amt = (weights*init_amt).rename(tkr_rename)
init_shares_contracts = (w_amt/init_px_amt).round(1)

# PFLIO :: POV EXPOSURES
ns = (init_shares_contracts).astype(int)
X = (ns*init_px_amt)/(ns*init_px_amt).sum()
X*S.rename(tkr_rename)
S*R
X*(S*R).rename(tkr_rename)

# PFL ASSESSMENT
roll_ret = ret.iloc[window: window+63].rename(columns=tkr_rename).copy()
pfl_ret_3M = roll_ret @ X
pfl_risk_3M = pfl_ret_3M.std()
# VIA positions
roll_px = price_wide.loc[roll_ret.index].rename(columns=tkr_rename).copy()
ns_mult = pd.Series([FUTURES_SPECS.get(tk)['multiplier'] for tk in roll_px.columns], index=roll_px.columns)
roll_init_pos = ns*init_px_amt
pfl_pnl_3M = (roll_px * ns * ns_mult).diff()
pfl_pnl_3M.iloc[0] = (roll_px * ns * ns_mult).iloc[0] - roll_init_pos
pfl_ret_3M_byPos = pfl_pnl_3M/init_amt
pfl_cret_3M_byPos = (100*((1+pfl_ret_3M_byPos).cumprod()-1)).round(2)
pfl_cret_3M_byPos.plot(); plt.show()

#%% MARKET DATA
prices = {
    "10YV25": 4.122,
    "6BZ25": 1.3483,
    "CLZ25": 60.36,
    "M6EZ25": 1.1789,
    "METV25": 4566.00,
    "ZBZ25": 116 + 29/32,
    "ZTZ25": 104+7.375/32,
    "MUB": 106.38,
    "NVDA": 187.56,
    "OKLO": 126.89,
    "PDBC": 13.35,
    "PLTM": 15.47,
    "TBT": 33.68,
    "URA": 49.60,
    "VGIT": 59.96,
    "WMT": 102.05,
    "XLRE": 42.08,
    "IAU": 73.22,
    "LRCX": 145.81,
    "FUUFF": 0.1277,
    "D": 61.53,
    "COPX": 61.97,
    "BEAM": 25.75,
    "ACRFF": 47.28
}

vols = {"MUB":0.0987,
        "6BZ25":0.0823,
        "CLZ25":0.5338,
        "ZBZ25":0.1393,
        "ZTZ25":0.0165}

unit_prices_options = {
    "6BZ25_DEC25_1.295P": 0.00260,
    "CLZ25_NOV25_90C": 0.090,
    "ZBZ25_NOV25_106P": 3/64,
    "ZTZ25_NOV25_105.25C": 0.75/32,
    "MUB_NOV25_103P": 0.350
}

#%% LOAD POSITIONS
fname_pfl = "portfolio.json"
path2pfl = os.path.join(os.environ["DATA_ROOT"], fname_pfl)
if not Path(path2pfl).exists():
    raise FileNotFoundError(f"Portfolio file not found: {path2pfl}")
with open(path2pfl, "r") as f:
    positions = json.load(f)
#%% PORTFOLIO
pm = PortfolioManager(positions=positions, futures_specs=FUTURES_SPECS)
#%% PORTFOLIO UPDATES
from portfolio_module import set_global_short_rate, get_global_short_rate
set_global_short_rate(0.042)  # affects mark_to_market and bumped_mtm defaults

updates = [
    {"symbol": "6BZ25", "type": "option", "expiry_days": 63},
    {"symbol": "MUB", "type": "option", "expiry_days": 49, "q":0.0313},
    {"symbol": "CLZ25", "type": "option", "expiry_days": 45},
    {"symbol": "ZBZ25", "type": "option", "expiry_days": 49},
    {"symbol": "ZTZ25", "type": "option", "expiry_days": 49}
]
n, details = pm.update_positions(updates)
print(n)
for d in details: print(d)
#%% PFOLIO M2M
total, rows, views = pm.mark_to_market_views(prices=prices, vols=vols, unit_prices_options=unit_prices_options)
df_m2m_class = pd.DataFrame(views["by_asset_class"]).T
df_m2m_equ_der = pd.DataFrame(views["equities_vs_derivatives"]).T
print(f"Total market value: {total:,.2f}")
print(f"\tBy asset class:\n", df_m2m_class.round(2))
print(f"\tEquities vs derivatives:\n", df_m2m_equ_der.round(2))
print()

# Inspect option validation (unit vs model)
for r in rows:
    if r.get("type") == "option":
        v = r.get("validation", {})  # our module stores unit/model comparison under "validation"
        print("symbol:", r.get("symbol"),
              "unit_provided:", v.get("unit_price_provided"),
              "model_unit:", v.get("model_unit_price"),
              "diff:", v.get("unit_vs_model_diff"),
              "pct:", v.get("unit_vs_model_pct"))

#%% MANUAL CHECK ON EQUITIES M2M
equities_mktval = {}
for pos in positions:
    if pos['type'] != 'equity':
        continue
    else:
        px_mark = prices.get(pos['symbol'])
        pos_mktval = px_mark * pos['quantity']
        equities_mktval[pos['symbol']] = pos_mktval
pfl_equities_mktval = np.sum((list(equities_mktval.values())))
print(f"{pfl_equities_mktval:,.2f}")

#%% MANUAL CHECK ON DERIVATIVES
from pricing_module import unit_price_option, tick_round
from portfolio_module import _opt_key_from_position
derivs_mktval = {}
for pos in positions:
    isDer = pos['type']=='future' or pos['type']=='option'
    if not isDer:
        continue
    else:
        postyp = pos['type']
        possym = pos.get('symbol')
        posqty = pos.get('quantity')
        posp0 = pos.get('trade_price')
        if postyp == 'future':
            mult = FUTURES_SPECS.get(possym).get('multiplier')
            posval = (prices.get(possym) - posp0)*posqty*mult
        else:
            K = pos.get('strike')
            T = pos.get('expiry_days')/365
            cp = pos.get('option_type')
            ivol =  vols.get(possym)
            opttyp = pos.get('underlying_type')
            if opttyp == 'future':
                und_sym = pos.get("underlying_symbol", possym)
                mult = FUTURES_SPECS.get(und_sym).get('multiplier')
                F_mark = prices.get(und_sym, prices.get(possym))
                pr_unit = unit_price_option(model="black",
                                            F=F_mark,
                                            K=K,
                                            T=T,
                                            r=get_global_short_rate(),
                                            sigma=ivol,
                                            option_type=cp)
                tick_size = FUTURES_SPECS.get(possym).get('tick_size')
            else:
                S = prices.get(pos.get("underlying_symbol", possym), prices.get(possym))
                mult = 100
                pr_unit = unit_price_option(model="bs",
                                            S=S,
                                            K=K,
                                            T=T,
                                            r=get_global_short_rate(),
                                            sigma=ivol,
                                            q=pos.get("q", 0.0313),
                                            option_type=cp)
                tick_size = 0.01

            pos_unit_mrktval = tick_round(float(pr_unit["price"]), float(tick_size))
            posval = pos_unit_mrktval*posqty*mult
            # Comparison
            unit_key = _opt_key_from_position(pos)
            market_unit_price = unit_prices_options[unit_key]
            unit_pr_miss = tick_round(market_unit_price-pos_unit_mrktval,
                                      float(tick_size))
            print(f"{unit_key}")
            print(f"\tMarket mismatch: {unit_pr_miss}")

        derivs_mktval[possym] = posval
pfl_derivs_mktval = np.sum((list(derivs_mktval.values())))
print(f"{pfl_derivs_mktval:,.2f}")


#%% TOTAL PNL BREAKDOWN
pfl_tpnl = 0
pfl_equ = 0
pfl_dervis = 0
for r in rows:
    pfl_tpnl += float(r.get('mtm_pnl'))
    if r.get("type") == "equity":
        pfl_equ += float(r.get('mtm_pnl'))
    else:
        pfl_dervis += float(r.get('mtm_pnl'))

print(f"Portfolio Total PnL: {pfl_tpnl:,.2f}")
print(f"\tEquities PnL: {pfl_equ:,.2f}")
print(f"\tDerivatives Total PnL: {pfl_dervis:,.2f}")

#%% TOTAL PNL BRKDN CHECKUP


#%% UNITTESTS (updated to use pricing_module.unit_price_option + apply_multiplier)
# ---------------------------------------------------------------------------------
print("\nUNITTESTS:")

# Example 1: price an equity call (NVDA)
unit_res = unit_price_option(model="bs", S=160.0, K=150.0, T=30/365, r=0.02, sigma=0.35, option_type="call")
print("BS unit price (NVDA 160/150):", unit_res.get("price"), "delta:", unit_res.get("delta"), "gamma:", unit_res.get("gamma"))
# apply multiplier (equity option typical multiplier = 100)
contract_cash = apply_multiplier(unit_res.get("price"), "NVDA_OPTION", futures_specs=FUTURES_SPECS)  # will default to 1 if not present
print("Contract-level cash (NVDA option, using default multiplier if absent):", contract_cash)

# Example 2: price CL futures option (Black)
unit_res2 = unit_price_option(model="black", F=61.37, K=90.0, T=47/365, r=0.04125, sigma=0.5225, option_type="call")
cl_multiplier = FUTURES_SPECS.get("CL", {}).get("multiplier", 1000)
contract_cl_cash = apply_multiplier(unit_res2.get("price"), "CL", futures_specs=FUTURES_SPECS)
print("Black unit price (CL):", unit_res2.get("price"), "delta:", unit_res2.get("delta"), "contract cash:", contract_cl_cash, "(multiplier:", cl_multiplier, ")")

# Bond future example (ZT)
tss2d = dict(zip([1,2,3,5,6,7,8,10], [0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]))
F = 104 + (9 + tss2d.get(7, 0))/32
K = 105.25
T = 50/365
r = 0.042
impvol = 0.017
cp = "call"
# unit price for ZT
unit_z = unit_price_option(model="black", F=F, K=K, T=T, r=r, sigma=impvol, option_type=cp)
print("ZT unit price:", unit_z.get("price"), "delta:", unit_z.get("delta"))

# 1-tick bump using tick from specs (1/256)
tick_ZT = FUTURES_SPECS.get("ZT", {}).get("tick_size", 1/256.0)
Fpx = F + tick_ZT
unit_z_bumped = unit_price_option(model="black", F=Fpx, K=K, T=T, r=r, sigma=impvol, option_type=cp)
print("ZT bumped unit price:", unit_z_bumped.get("price"))
print("Tick:", tick_ZT, "Delta-approx price change (unit-level):", tick_ZT * unit_z.get("delta"))
# contract-level delta-approx (apply multiplier)
zt_mult = FUTURES_SPECS.get("ZT", {}).get("multiplier", 2000)
print("Delta-approx cash change (contract-level):", tick_ZT * unit_z.get("delta") * zt_mult)

# FX example
F_fx, K_fx, T_fx, r_fx, impvol_fx, cp_fx = 1.3447, 1.295, 64/365, 0.042, 0.083, 'put'
unit_fx = unit_price_option(model="black", F=F_fx, K=K_fx, T=T_fx, r=r_fx, sigma=impvol_fx, option_type=cp_fx)
print("FX put unit price:", unit_fx.get("price"), "delta:", unit_fx.get("delta"))
delta_F = 0.0001
unit_fx_bumped = unit_price_option(model="black", F=F_fx + delta_F, K=K_fx, T=T_fx, r=r_fx, sigma=impvol_fx, option_type=cp_fx)
print("FX bumped unit price:", unit_fx_bumped.get("price"), "approx change (unit):", unit_fx_bumped.get("price") - unit_fx.get("price"))
# contract-level example for FX (use FUTURES_SPECS if you have a multiplier)
fx_mult = FUTURES_SPECS.get("6B", {}).get("multiplier", None)
if fx_mult:
    print("FX put contract-level cash:", apply_multiplier(unit_fx.get("price"), "6B", futures_specs=FUTURES_SPECS))

