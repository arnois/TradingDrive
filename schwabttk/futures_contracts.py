#%% LIBS
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from schwabttk.price_history import load_futures_symbols, load_stored
from schwabttk.visualize import plot_ohlc, plot_multi, plot_corr, show


#%% VIZ
# Single candlestick
fig, df = plot_ohlc("ZW", start="2017-01-01")
show(fig, "ZW", df=df)

# Compare a cross-asset subset normalized
fig = plot_multi(
    tickers=["M2K", "MGC", "MCL", "ZN", "6B", "MBT"],
    start="2023-01-01",
    theme='greenredoverwhite'
)
show(fig, "multi_normalized")

# Full universe correlation
fig = plot_corr(start="2023-01-01",theme='greenredoverwhite')
show(fig, "corr_heatmap")

#%% Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) # BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data", "futures")

#%% SYMBOLS
symbols = load_futures_symbols()
print(symbols)

#%% BY SYMBOL DATA
ssymb = 'M2K'
data = load_stored(ssymb)

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

