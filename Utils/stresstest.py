from sqlite3 import DatabaseError
import sys
sys.path.append("/Users/ZiyanLai/Dropbox/Files/Crypto")
import pandas as pd
import numpy as np
# from yahoo_fin import stock_info as si
from tqdm import tqdm
from math import exp, log, sqrt, floor
from datetime import datetime, timedelta 
from IPython.display import display
from Utils.strategy import EnhancedDeMark
from Utils.backtest import Backtester
from Utils.APIs import DataSourcer

# dpi = si.get_data("DPI-USD")["adjclose"]
# eth = si.get_data("ETH-USD")["adjclose"]
dpi = DataSourcer("dpi").get_daily_data()["prices"].resample("1d").last()
eth = DataSourcer("eth").get_daily_data()["prices"].resample("1d").last()

end = pd.to_datetime("today").replace(hour=0,minute=0,second=0,microsecond=0,nanosecond=0)
start = (end - timedelta(days=550)).replace(hour=0)
end = min(dpi.index.max(), eth.index.max(), end)
start = max(dpi.index.min(), eth.index.min(), start)
day_range = pd.date_range(start, end, freq="D")

num_paths = 2000

def max_drawdown(series):
    ret = np.log(series).diff().dropna()
    peak = bottom = max_dd = 0
    peakInd = bottomInd = 0
    for i in range(1, len(ret)-1):
        prev, curr, nxt = ret[i-1], ret[i], ret[i+1]
        if curr >= prev and curr >= nxt:
            peak = curr
            peakInd = i
        elif curr <= prev and curr <= nxt:
            bottom = curr
            bottomInd = i
        if peakInd < bottomInd:
            max_dd = max(max_dd, peak-bottom)
    return max_dd

def realized_vol(series):
    ret = np.log(series).diff()
    N = len(series)
    if N <= 2: return np.nan
    dt = (series.index[1]-series.index[0]).total_seconds() / (24*3600*365)
    T = N * dt
    vol = (np.square(ret).sum()/T) ** 0.5
    return vol

def compute_params(series, omega=None, window=90):
#     Et = dpi.resample("W-Wed").last().loc[:dpi.index.max()]
#     rVol = series.rolling("90D", min_periods=90).apply(realized_vol).bfill()
    rVol = series.rolling(f"{window}D", min_periods=window).apply(realized_vol).bfill().resample("W-Wed").last().loc[start:end]
    rVol = rVol.reindex(day_range).bfill().resample("1h").bfill().ffill()
    Einf = series.shift(-180).ffill().ewm(0.75).mean().resample("W-Wed").last().loc[start:end]
    Einf = Einf.reindex(day_range).bfill().resample("1h").bfill().ffill()
#     t = (Et.index-series.index.min()).days/365
#     M = (np.log(Et/dpi.loc[dpi.index.min()]) - (rVol.loc[Et.index]**2)*(1-np.exp(-2*omega*t))/(4*omega)) / (1-np.exp(-omega*t))
    if omega:
        M = (np.log(Einf/series.loc[start]) - (rVol.loc[Einf.index]**2)/(4*omega))
        M = M.reindex(day_range).bfill().resample("1h").bfill().ffill()
    else:
        M = pd.Series(np.nan, rVol.index)
    params = pd.concat([M, rVol, Einf], axis=1)
    params.columns = ["M","vol","Einf"]
    return params

def simulate_index(params, seed):
    np.random.seed(seed)
    z = np.random.normal(0,1,len(params))
    x0 = 0
    xt = [x0]
    t0 = start
    s0 = dpi.loc[t0]
    dts = np.diff((params.index-t0).total_seconds()/(365*24*3600))
    Ms, vols, omegas = params["M"], params["vol"], params["omega"]
    for i in range(len(params.index)):
        if i == 0: continue
        dt = dts[i-1]
        M, vol, omega = Ms[i], vols[i], omegas[i]
        x = xt[i-1]*exp(-omega*dt) + M*(1-exp(-omega*dt)) + sqrt(((1-exp(-2*omega*dt))*(vol**2))/2*omega)*z[i]
        x = max(log(50/s0), x)
        x = min(log(800/s0), x)
        xt.append(x)
    xt = pd.Series(xt, params.index)
    st = pd.Series(s0*np.exp(xt), params.index)
    sim = pd.concat([xt, st], axis=1)
    sim.columns = ["xt","st"]
    return sim

def simulate_index_paths(omega=0.8, window=90):
    seeds = np.arange(1,num_paths+1)
    params = compute_params(dpi, omega=omega, window=window)
    params["omega"] = omega
    sNod_sims = pd.DataFrame(columns=seeds)
    xNod_sims = pd.DataFrame(columns=seeds)
    for seed in tqdm(seeds, desc="Simulating benchmark paths"):
        sim = simulate_index(params, seed=seed)
        xNod_sims[seed] = sim["xt"]
        sNod_sims[seed] = sim["st"]
    return params, xNod_sims, sNod_sims

def simulate_individual(params, seed, xNod_dpi, s_dpi):
    np.random.seed(seed)
    z1 = np.random.normal(0,1,len(params))
    z2 = np.random.normal(0,1,len(params))
    poisson = np.random.poisson(lam=1/24, size=len(params))
    corrs, vols, omegas, c = params["corr"], params["vol"], params["omega"], params["c"]
    z = corrs*z1 + np.sqrt(1-np.square(corrs))*z2
    t0 = start
    s0 = eth.loc[t0]
#     xBar = np.log(s_dpi/s0) - np.square(vols)/(4*omegas)
    xBar = xNod_dpi + c
    x0 = 0
    x = [x0]
    dts = np.diff((params.index-t0).total_seconds()/(365*24*3600))
    for i in range(len(params.index)):
        if i == 0: continue
        dt = dts[i-1]
        currXBar, prevXBar, prevX = xBar[i], xBar[i-1], x[i-1]
        vol, omega = vols[i], omegas[i]
        currX = currXBar + exp(-omega*dt)*(prevX-prevXBar) + vol*sqrt((1-exp(-2*omega*dt))/(2*omega))*z[i]
        for _ in range(poisson[i]):
            jumpSize = np.random.normal(0,0.15)
            currX += jumpSize
        currX = max(log(300/s0), currX)
        currX = min(log(8000/s0), currX)
        x.append(currX)
    st = pd.Series(s0*np.exp(x), index=params.index)
    return st

def simulate_individual_paths(dpiParams, xNodSims, dpiSims, omega2=0.35, window=45):
    seeds = dpiSims.columns
    M, EnodInf, vol1, omega1 = dpiParams["M"], dpiParams["Einf"], dpiParams["vol"], dpiParams["omega"]
    tmpParams = compute_params(eth, window=window)
    vol2, Einf = tmpParams["vol"], tmpParams["Einf"]
    k = np.log(Einf/eth.loc[start]) / np.log(EnodInf/dpi.loc[start])
    c = (k-1) * (M + (vol1**2)/(4*omega1)) - (vol2**2)/(4*omega2)
    corr = dpi.rolling(f"{window}D", min_periods=window).corr(eth).bfill().resample("W-Wed").last().loc[start:end]
    corr = corr.reindex(day_range).bfill().resample("1h").bfill().ffill()
    simParams = pd.concat([vol2,corr,c], axis=1)
    simParams.columns = ["vol","corr","c"]
    simParams["omega"] = omega2
    sims = pd.DataFrame(columns=seeds)
    for seed in tqdm(seeds, desc="Simulating crypto paths"):
        sims[seed] = simulate_individual(simParams, seed, xNodSims[seed], dpiSims[seed])
    return sims

def expected_extreme_value(series, upOrDown):
    ascending = False if upOrDown=="up" else True
    sortedVal = series.sort_values(ascending=ascending)
    ind = floor(len(sortedVal)*0.1)
    return sortedVal.iloc[:ind].mean()

def get_scenarios(sims):
    def cum_ret(series):
        return np.log(series.loc[series.index.max()]/series.loc[series.index.min()])
    
    info = []
    for i in tqdm(sims.columns, desc="Generating top scenarios"):
        maxVol = sims[i].resample("10d").apply(realized_vol).max()
        info.append([i, maxVol])
    volPaths = sorted(info, key=lambda x:x[1], reverse=True)[:100]
    stablePaths = sorted(info, key=lambda x:x[1], reverse=False)[:100]
    volSims, stableSims = sims[[x[0] for x in volPaths]], sims[[x[0] for x in stablePaths]]
    volUp, volDown = [], []
    stableUp, stableDown = [], []
    
    for i in volSims.columns:
        cumRet = volSims[i].resample("10d").apply(cum_ret)
        volUp.append([i, expected_extreme_value(cumRet, "up")])
        volDown.append([i, expected_extreme_value(cumRet, "down")])
    volUp = sorted(volUp, key=lambda x:x[1], reverse=True)[:20]
    volDown = sorted(volDown, key=lambda x:x[1], reverse=False)[:20]
    for i in stableSims.columns:
        cumRet = stableSims[i].resample("10d").apply(cum_ret)
        stableUp.append([i, expected_extreme_value(cumRet, "up")])
        stableDown.append([i, expected_extreme_value(cumRet, "down")])
    stableUp = sorted(stableUp, key=lambda x:x[1], reverse=True)[:20]
    stableDown = sorted(stableDown, key=lambda x:x[1], reverse=False)[:20]
    scenarios = pd.DataFrame(columns=["Vol Up", "Vol Down", "Stable Up", "Stable Down"])
    scenarios["Vol Up"] = [x[0] for x in volUp]
    scenarios["Vol Down"] = [x[0] for x in volDown]
    scenarios["Stable Up"] = [x[0] for x in stableUp]
    scenarios["Stable Down"] = [x[0] for x in stableDown]
    return scenarios

def scenario_table(scenarios, dpiSims, ethSims):
    multiIndex = [('Return', '0.25Q'),
                ('Return', '0.5Q'),
                ('Return', '0.75Q'),
                ('Return', 'Mean'),
                ('Max Drawdown', '0.25Q'),
                ('Max Drawdown', '0.75Q'),
                ('Max Drawdown', 'Mean'),
                ('Max Drawdown', 'Max'),
                ('Sortino Ratio', 'Min',),
                ('Sortino Ratio', 'Max',),
                ('Sortino Ratio', 'Mean',),
                ('Downside Beta', 'Min',),
                ('Downside Beta', 'Max',),
                ('Downside Beta', 'Mean',),
                ('Capital Usage', 'Min'),
                ('Capital Usage', 'Max'),
                ('Capital Usage', 'Mean')]
    index = pd.MultiIndex.from_tuples(multiIndex)
    stressRes = pd.DataFrame(columns=scenarios.columns, index=index)
    for scenario in scenarios.columns:
        returns, sortinos, downsideBetas, maxDrawdowns, capitals = [],[],[],[],[]
        for path in tqdm(scenarios[scenario], desc=f"Running scenario {scenario}"):
            hourly_data = pd.DataFrame(ethSims[path])
            hourly_data.columns = ["close"]
            indexSim = dpiSims[path].resample("1d").mean()
            backtester = Backtester(Strategy=EnhancedDeMark, hourly_data=hourly_data, days=365, benchmark=indexSim)
            backtester.tqdm_disable = True
            backtester.base_backtest()
            returns.append(backtester.tracker["return"].cumsum()[-1])
            sortinos.append(backtester.sortino_ratio(backtester.tracker))
            downsideBetas.append(backtester.downside_beta(backtester.tracker))
            maxDrawdowns.append(backtester.max_drawdown(backtester.tracker))
            capitals.append(backtester.capital)
        stressRes.loc[("Return","0.25Q"), scenario] = np.quantile(returns, 0.25)
        stressRes.loc[("Return","0.5Q"), scenario] = np.quantile(returns, 0.5)
        stressRes.loc[("Return","0.75Q"), scenario] = np.quantile(returns, 0.75)
        stressRes.loc[("Return","Mean"), scenario] = np.mean(returns)
        stressRes.loc[("Max Drawdown","0.25Q"), scenario] = np.quantile(maxDrawdowns, 0.25)
        stressRes.loc[("Max Drawdown","0.75Q"), scenario] = np.quantile(maxDrawdowns, 0.75)
        stressRes.loc[("Max Drawdown","Mean"), scenario] = np.mean(maxDrawdowns)
        stressRes.loc[("Max Drawdown","Max"), scenario] = np.max(maxDrawdowns)
        stressRes.loc[("Sortino Ratio","Min"), scenario] = np.min(sortinos)
        stressRes.loc[("Sortino Ratio","Max"), scenario] = np.max(sortinos)
        stressRes.loc[("Sortino Ratio","Mean"), scenario] = np.mean(sortinos)
        stressRes.loc[("Downside Beta","Min"), scenario] = np.min(downsideBetas)
        stressRes.loc[("Downside Beta","Max"), scenario] = np.max(downsideBetas)
        stressRes.loc[("Downside Beta","Mean"), scenario] = np.mean(downsideBetas)
        stressRes.loc[("Capital Usage","Min"), scenario] = np.min(capitals)
        stressRes.loc[("Capital Usage","Max"), scenario] = np.max(capitals)
        stressRes.loc[("Capital Usage","Mean"), scenario] = np.mean(capitals)
    return stressRes

if __name__ == '__main__':
	dpiParams, xNodSims, dpiSims = simulate_index_paths()
	ethSims = simulate_individual_paths(dpiParams, xNodSims, dpiSims)
	scenarios = get_scenarios(ethSims)
	stressRes = scenario_table(scenarios, dpiSims, ethSims)
	stressRes.to_excel(f"stress result_{datetime.today().strftime('%Y-%m-%d')}.xlsx")
	display(stressRes)