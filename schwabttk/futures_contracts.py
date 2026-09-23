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
    tickers=["M2K", "MGC", "MCL", "ZN", "6B", "ZW"],
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

