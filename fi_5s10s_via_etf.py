#%% LIBS
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#from matplotlib.ticker import FuncFormatter
#%% PLOTTING AES
plt.rcParams['figure.dpi'] = 72.0
plt.rcParams['figure.figsize'] = [6, 4]
Colors = ['#0626a9', '#ffc62d', '#f08900', '#e44261', '#00c4b3', '#6c7bd3', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color = Colors)
#%% PATHS
datapath = r'C:\dev\data\squests'

#%% DATA IMPORT
treasury_files = glob.glob(os.path.join(datapath,"daily-treasury-rates*.csv"))
dfs = []
for f in treasury_files:
    df = pd.read_csv(f, parse_dates=["Date"])
    dfs.append(df)

# Concatenate and drop duplicates
ust_yields = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Date"])
ust_yields = ust_yields.set_index("Date").sort_index()

# Load ETF data
etf_data = pd.read_csv(
    os.path.join(datapath, "data_PRC_EQU.csv"),
    parse_dates=["TRADING_DATE"],
    date_format="%m/%d/%Y %I:%M:%S %p"  # matches "12/31/2009 12:00:00 AM"
)
etf_data_C = etf_data.pivot(index="TRADING_DATE", columns="RIC", values="LAST")
etf_data_C.index.name = ust_yields.index.name

# Merge datasets on date
merged = ust_yields.join(etf_data_C, how="inner")  # or "outer" if you want all dates

# Clean column names
sel_assets = ["UST_5Y","UST_30Y","TBT","VGIT"]
data = merged.rename(columns={
    "5 Yr": "UST_5Y",
    "30 Yr": "UST_30Y",
    "TBT": "TBT",
    "VGIT": "VGIT",
    "VGIT.OQ": "VGIT"
})[sel_assets]

#%% DATA MGMT
data = data.interpolate(method='time')
data['UST_5s30s'] = (data['UST_30Y'] - data['UST_5Y'])*100
data['PFL_5s30s'] = data['TBT'] + data['VGIT']*4
data['PFL_5s30s'].loc[:'2012-10-04'] = data['PFL_5s30s'].loc[:'2012-10-04']+48.56
#%% LEVELS
fig, ax1 = plt.subplots(figsize=(10, 6))

# Primary axis (UST)
ax1.plot(data["UST_5s30s"], color=Colors[0], alpha=0.8, label="UST 5s30s")
ax1.set_ylabel("UST 5s30s (bp)", color=Colors[0], fontsize=12)
ax1.grid(False)

# Secondary axis (PFL)
ax2 = ax1.twinx()
ax2.plot(data["PFL_5s30s"], color=Colors[2], alpha=0.9, label="TBT+VGIT PFL")
ax2.set_ylabel("Portfolio ($-Value)", color=Colors[2], fontsize=12)

# Title and layout
ax1.set_title("UST 5s30s vs. Proxy Portfolio", fontsize=20)
ax1.set_xlabel("")

# Handle legends (combine from both axes)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, title="", loc="best")

#plt.tight_layout()
plt.show()

#%% RETURNS
pfolio_cols = ['UST_5s30s','PFL_5s30s']
df_ret = data[pfolio_cols].diff().dropna()
df_ret = df_ret.clip(
    lower={"UST_5s30s": -21, "PFL_5s30s": -3},
    upper={"UST_5s30s": 21,  "PFL_5s30s":  3}
)
#%% RETURNS :: SCATTER PLOT
df_plot = df_ret.clip(
    lower={"UST_5s30s": -21, "PFL_5s30s": -3},
    upper={"UST_5s30s": 21,  "PFL_5s30s":  3}
)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_plot["UST_5s30s"], df_plot["PFL_5s30s"], color = Colors[0], alpha=0.6)
# Diagonal reference line
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),
    np.max([ax.get_xlim(), ax.get_ylim()])
]
ax.plot(lims, lims, '--', linewidth=1, label='y = x', color=Colors[3])
# Final formatting
ax.set_xlim(lims)
ax.set_ylim(df_plot.min()['PFL_5s30s'], df_plot.max()['PFL_5s30s'])
ax.set_xlabel("Δ UST 5s30s (bp)")
ax.set_ylabel("Portfolio Return ($)")
ax.set_title("UST 5s30s vs. ETF Proxy Portfolio", fontsize=18)
ax.legend(title="")
ax.grid(False)
plt.tight_layout()
plt.show()

#%% PFL BETA VS UST 5s30s
X = df_ret["UST_5s30s"]   # slope changes in bps
y = df_ret["PFL_5s30s"]   # portfolio P&L in $

X = sm.add_constant(X)    # add intercept
model = sm.OLS(y, X).fit()

print(model.summary())

#%% PFL BETA TO UST 5s30s PLOT
x = df_ret["UST_5s30s"]
y = df_ret["PFL_5s30s"]

m, b = np.polyfit(x, y, 1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, alpha=0.5)
ax.plot(x, m*x + b, color=Colors[2], label=f"Beta (reg) = {m:.3f}")
ax.axhline(0, color="grey", linestyle="--", alpha=0.2)
ax.axvline(0, color="grey", linestyle="--", alpha=0.2)
ax.set_xlabel("Δ5s30s (bps)")
ax.set_ylabel("Portfolio Return ($)")
ax.legend()
ax.set_title("Portfolio P&L vs Δ5s30s ", fontsize=18)
plt.tight_layout()
plt.show()

#%% PFL BETA STABILITY
theoretical_beta = 0.1184
window = 64 # 3M
betas = []
for i in range(window, len(df_ret)+1):
    sub = df_ret.iloc[i-window:i]  # rolling window
    model = sm.OLS(sub["PFL_5s30s"],
                   sm.add_constant(sub["UST_5s30s"])
                   ).fit()
    betas.append(model.params[1])

betas = pd.Series(betas, index=df_ret.index[window-1:], name="Beta")
#%% PFL BETA STABILITY :: PLOT
fig, ax = plt.subplots(figsize=(10,6))
betas.plot(color=Colors[0])
ax.axhline(theoretical_beta,
            color=Colors[1],
            linestyle="--",
            label=f"Beta (Theory) = {theoretical_beta:.2f}")
ax.set_xlabel("")
ax.set_ylabel("Beta")
ax.legend()
ax.set_title(f"Rolling {window}-D Hedge Ratio ($ per bp)", fontsize=18)
plt.tight_layout()
plt.show()

#%% SLOPE REGIMES
epsilon = 1.5  # threshold in bps

def classify_regime(change):
    if change > epsilon:
        return "Steepening"
    elif change < -epsilon:
        return "Flattening"
    else:
        return "Stable"

df_ret["Regime"] = df_ret["UST_5s30s"].apply(classify_regime)

#%% PFL BETA WITHIN REGIME
results = {}
for regime, sub in df_ret.groupby("Regime"):
    X = sm.add_constant(sub["UST_5s30s"])
    y = sub["PFL_5s30s"]
    model = sm.OLS(y, X).fit()
    results[regime] = {
        "Beta": model.params[1],
        "R2": model.rsquared,
        "N": len(sub)
    }
print(pd.DataFrame(results).T)
#%% PFL BETA WITHIN REGIME :: BOXPLOTS
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(x="Regime", y="PFL_5s30s", data=df_ret, color=Colors[0],)
plt.axhline(0, color="grey", linestyle="--", alpha=0.2)
plt.title("Portfolio Returns by 5s30s Regime",fontsize=18)
plt.show()

#%% PFL BETA BY REGIME
betas = {k: v["Beta"] for k, v in results.items()}
fig, ax = plt.subplots(figsize=(10,6))
plt.bar(betas.keys(), betas.values(), color=Colors[4], label="Empirical")
plt.axhline(0.12, color="red", linestyle="--", label="Theoretical")
plt.ylabel("Beta ($ per bp)")
plt.title("Empirical Hedge Ratio by 5s30s Regime",fontsize=18)
plt.legend(title="", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% PFL ROLLING BETA BY REGIME
window = 64  #
lst_regime = df_ret['Regime'].unique()
lst_betas_regime = []
for regime in lst_regime:
    df_ret_selregime = df_ret[df_ret['Regime'] == regime].copy()
    betas_regime= []
    for i in range(window, len(df_ret_selregime) + 1):
        sub = df_ret_selregime.iloc[i - window:i]  # rolling window
        model = sm.OLS(sub["PFL_5s30s"],
                       sm.add_constant(sub["UST_5s30s"])
                       ).fit()
        betas_regime.append(model.params[1])

    lst_betas_regime.append(pd.Series(betas_regime,
                            index=df_ret_selregime.index[window - 1:],
                            name=f"Beta_{regime}"))
#%%
df_betas_regime = pd.DataFrame(lst_betas_regime).T
#%% PLOT
fig, ax = plt.subplots(figsize=(10,6))
df_betas_regime.interpolate(method='time').plot()
ax.axhline(theoretical_beta,
            color=Colors[1],
            linestyle="--",
            label=f"Beta (Theory) = {theoretical_beta:.2f}")
ax.set_xlabel("")
ax.set_ylabel("Beta")
ax.legend(title="",bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title(f"Rolling {window}-D Hedge Ratio ($ per bp)", fontsize=18)
plt.tight_layout()
plt.show()