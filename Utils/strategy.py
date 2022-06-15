from signal import signal
import pandas as pd
import numpy as np
import scipy.stats as st
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from scipy.integrate import quad
from pdb import set_trace as bp

class Indicator(ABC):
    def __init__(self, hourly_data, start=None, end=None, days=None):
        self.hourly_data = hourly_data
        self.daily_data = self.__aggregate_daily_ohlc_vwap(self.hourly_data)

    def __vwap(self, df, column):
        return (df[column]*df["Volume USD"]).sum() / df["Volume USD"].sum()
        
    def __aggregate_daily_ohlc_vwap(self, df):
        if df.shape[1] != 1:
            close = df.resample("1d").apply(self.__vwap, column="close")
            high = df.resample("1d").apply(self.__vwap, column="high")
            low = df.resample("1d").apply(self.__vwap, column="low")
            volume = df["Volume USD"].resample("1d").sum()
        else:
            close = df["close"].resample("1d").last()
            high = df["close"].resample("1d").max()  
            low = df["close"].resample("1d").min()      
            volume = pd.Series(1, index=close.index)
        df = pd.DataFrame({"close":close, "high":high, "low":low, "volume":volume}, index=close.index)
        return df
    
    @abstractmethod
    def generate_signal():
        pass
    @abstractmethod
    def plot_signal():
        pass

class OnBalanceVolume(Indicator):
    def __init__(self, hourly_data):
        super().__init__(hourly_data)
        self.signal = pd.DataFrame(columns=["OBV"], dtype="float64")
        
    def generate_signal(self, i, t):
        price = self.daily_data["close"]
        volume = self.daily_data["volume"]
        if i == 0:
            self.signal.loc[t,"OBV"] = volume.iat[0]
            return
        price_today = price.iat[i]
        price_yesterday = price.iat[i-1]
        obv_yesterday = self.signal.iloc[i-1]["OBV"]
        if price_today > price_yesterday:
            self.signal.loc[t,"OBV"] = obv_yesterday + volume.iat[i]
        elif price_today < price_yesterday:
            self.signal.loc[t,"OBV"] = obv_yesterday - volume.iat[i]
        else:
            self.signal.loc[t,"OBV"] = obv_yesterday
        self.signal["delta OBV"] = self.signal["OBV"].diff(1)
    
    def plot_signal(self, ax, start, end):
        signal = self.signal.loc[start:end]
        ax2 = ax.twinx()
        lines = ax.plot(signal["OBV"], "--", label="OBV", alpha=0.8)
        lines += ax2.plot(signal["delta OBV"], c="darkorange", label="$\Delta^1$ OBV")
        ax.legend(lines, [l.get_label() for l in lines], loc=2)
        ax2.grid(False)
        return ax

class BollingerBands(Indicator):
    def __init__(self, hourly_data, alpha, lookback_days, k, j, big_CI=0.85, small_CI=0.65, estimator="TSRV"):
        super().__init__(hourly_data)
        self.k = k
        self.j = j
        self.lookback_days = lookback_days
        self.big_CI, self.small_CI = big_CI, small_CI
        self.big_up_z, self.small_up_z = st.norm.ppf(big_CI+(1-big_CI)/2), st.norm.ppf(small_CI+(1-small_CI)/2)
        self.big_down_z, self.small_down_z = st.norm.ppf((1-big_CI)/2), st.norm.ppf((1-small_CI)/2)
        self.alpha = alpha
        self.estimator = estimator
        self.signal = pd.DataFrame(columns=["EMA", "std", "kurt", "BigUp", "SmallUp", "BigDown", "SmallDown"], dtype=np.float64)
        if self.hourly_data.shape[1] != 1:
            self.hourly_price = (self.hourly_data["low"] + self.hourly_data["high"] + self.hourly_data["close"]) / 3
        else:
            self.hourly_price = self.hourly_data["close"]
        self.daily_price = (self.daily_data["low"] + self.daily_data["high"] + self.daily_data["close"]) / 3
        
    def __TSRV(self, data):
        data = np.log(data)
        k, j = self.k, self.j
        n = len(data)
        nbarK, nbarJ = (n-k+1)/k, (n-j+1)/j
        adj = (1-(nbarK/nbarJ))**(-1)
        RV_k = np.square(data - data.shift(k)).sum() / k
        RV_j = np.square(data - data.shift(j)).sum() / j
        RV = adj * (RV_k - (nbarK/nbarJ) * RV_j)
        sqrt = np.sqrt(max(0, RV))
        return sqrt
    
    def __realized_moments(self, data):
        data = np.log(data)
        g = lambda x: min(x, 1-x)
        g2 = lambda x: pow(g(x),2)
        g3 = lambda x: pow(g(x),2)
        g4 = lambda x: pow(g(x),4)
        g2bar = quad(g2,0,1)[0]
        g3bar = quad(g3,0,1)[0]
        g4bar = quad(g4,0,1)[0]
        k = self.k
        def f(grid):
            res = 0
            ret = grid.diff(1)
            for i in range(1,len(ret)):
                res += g(i/k)*ret[i]
            return res
        def f_bar(grid):
            res = 0
            ret = grid.diff(1)
            for i in range(1,len(ret)):
                res += ((g(i/k)-g((i-1)/k))*ret[i])**2
            return res

        deltaY = data.rolling(window=k).apply(f).dropna()
        deltaYBar = data.rolling(window=k).apply(f_bar).dropna()

        realizedKurt = (deltaY**4).sum() / (k*g4bar)
        realizedSkew = (deltaY**3).sum() / (k*g3bar)
        realizedVar = ((deltaY**2).sum()/k - deltaYBar.sum()/(2*k))/g2bar
        return realizedKurt/(realizedVar**2), realizedSkew/(realizedVar**(3/2))
    
    def generate_signal(self, t):
        end_hour = t.replace(hour=23)
        tsrv_start_hour = end_hour - timedelta(days=self.lookback_days) + timedelta(hours=1)
        kurt_start_hour = end_hour - timedelta(days=self.lookback_days) + timedelta(hours=1)
        ema = self.daily_price.loc[:t].ewm(alpha=self.alpha).mean().loc[t]
        kurt, skew = self.__realized_moments(self.hourly_price.loc[kurt_start_hour:end_hour])
        if self.estimator == "TSRV":
            tsrv = self.__TSRV(self.hourly_price.loc[tsrv_start_hour:end_hour])
            std = tsrv / np.sqrt(self.lookback_days)
            big_up, small_up = ema * (1+self.big_up_z*std), ema * (1+self.small_up_z*std)
            big_down, small_down = ema * (1+self.big_down_z*std), ema * (1+self.small_down_z*std)
        if self.estimator == "simple":
            std = self.daily_tp.ewm(alpha=self.alpha).std()
            big_up, small_up = ema + 2*std, ema + 1*std
            big_down, small_down = ema - 2*std, ema - 1*std
        self.signal.loc[t,"EMA"] = ema
        self.signal.loc[t,"BigUp"] = big_up
        self.signal.loc[t,"std"] = std
        self.signal.loc[t,"BigDown"] = big_down
        self.signal.loc[t,"SmallUp"] = small_up
        self.signal.loc[t,"SmallDown"] = small_down
        self.signal.loc[t,"kurt"] = kurt
        self.signal.loc[t,"skew"] = skew
        
    def plot_signal(self, ax1, ax2, start, end):
        signal = self.signal.loc[start:end]
        ema, bigUp, smallUp, bigDown, smallDown, kurt, skew = signal["EMA"], signal["BigUp"],\
                                                        signal["SmallUp"], signal["BigDown"], signal["SmallDown"], signal["kurt"], signal["skew"]
        ax1.plot(ema, "--", color="darkred", label="EMA", alpha=0.8)        
        ax1.plot(bigUp, color="darkred", label="{:.0%} C.I.".format(self.big_CI), alpha=0.5)
        ax1.plot(smallUp, color="darkgrey", label="{:.0%} C.I.".format(self.small_CI))
        ax1.plot(smallDown, color="darkgrey")
        ax1.plot(bigDown, color="darkred", alpha=0.5)
        ax1.fill_between(smallUp.index, smallUp, bigUp, color="darkorange", alpha=0.4)
        ax1.fill_between(smallDown.index, smallDown, bigDown, color="darkorange", alpha=0.4)
        ax2.plot(kurt, label=f"{self.lookback_days}-day Kurt")
        ax2.plot(skew, label=f"{self.lookback_days}-day Skew")
        ax1.legend(loc=2)
        ax2.legend(loc=2)
        return ax1, ax2

class DeMarkSequence(Indicator):
    def __init__(self, hourly_data=None, 
                       setup_lookback=4, 
                       setup_count=9, 
                       buy_countdown_period=13, 
                       sell_countdown_period=13, 
                       notional=10000):
        super().__init__(hourly_data)
        self.obv = OnBalanceVolume(hourly_data)
        self.univ_sell_countdown_period = self.sell_countdown_period = sell_countdown_period
        self.univ_buy_countdown_period = self.buy_countdown_period = buy_countdown_period
        self.setup_count = setup_count
        self.setup_lookback = setup_lookback
        self.notional = notional
        self.signal = pd.DataFrame(columns=["buy","sell","buy setup","sell setup","buy countdown","sell countdown"])
        self.reset("buy")
        self.reset("sell")
        
    def reset(self, side):
        if side == "buy":
            self.countdownBuy, self.isSetupBuy, self.setupIndBuy = 0, False, None
        elif side == "sell":
            self.countdownSell, self.isSetupSell, self.setupIndSell = 0, False, None
        
    def setup(self, setupInd, t, side, status, countdown_period):
        if side == "buy":
            self.isSetupBuy, self.setupIndBuy = True, setupInd
            self.signal.loc[t, "buy setup"] = status
            self.buy_countdown_period = countdown_period
        elif side == "sell":
            self.isSetupSell, self.setupIndSell = True, setupInd
            self.signal.loc[t, "sell setup"] = status
            self.sell_countdown_period = countdown_period
    
    def __recycle(self, setupInd, t, side):
        self.reset(side)
        if side == "buy":
            self.setup(setupInd, t, side, "R", self.univ_buy_countdown_period)
        elif side == "sell":
            self.setup(setupInd, t, side, "R", self.univ_sell_countdown_period)

    def cancel_setup(self, t, side): 
        self.reset(side)
        self.signal.loc[t, f"{side} setup"] = "X"
        
    def __true_high_and_low(self, window_start, window_end):
        sub = self.daily_data.iloc[window_start:window_end]
        return sub["high"].max(), sub["low"].min()
    
    def check_setup(self, i, side, count=None, lookback=None):
        if not count: count = self.setup_count
        if not lookback: lookback = self.setup_lookback
        setup_start, setup_end = i-count, i
        if setup_start-lookback < 0: return False    
        window = self.daily_data.iloc[setup_start:setup_end]
        lag_window = self.daily_data.iloc[setup_start-lookback:setup_end-lookback]
        if side == "buy":
            return sum(window["close"].values < lag_window["close"].values) == count
        elif side == "sell":
            return sum(window["close"].values > lag_window["close"].values) == count

    def __check_countdown(self, i, t, side, lookback=2):
        if side == "buy":
            if self.daily_data.iloc[i]["close"] < self.daily_data.iloc[i-lookback]["low"]:
                self.countdownBuy += 1
                self.signal.loc[t,"buy countdown"] = self.countdownBuy
            elif self.countdownBuy >= self.buy_countdown_period:
                self.signal.loc[t,"buy countdown"] = self.countdownBuy
            complete = (self.countdownBuy >= self.buy_countdown_period)
        elif side == "sell":
            if self.daily_data.iloc[i]["close"] > self.daily_data.iloc[i-lookback]["high"]:
                self.countdownSell += 1
                self.signal.loc[t,"sell countdown"] = self.countdownSell
            elif self.countdownSell >= self.sell_countdown_period:
                self.signal.loc[t,"sell countdown"] = self.countdownSell
            complete = (self.countdownSell >= self.sell_countdown_period)
        if complete:
            self.reset(side)
            self.signal.loc[t, side] = 1
    # Think about when selling
    def __check_recycle(self, new_setup_ind, old_setup_ind):
        count, lookback = self.setup_count, self.setup_lookback
        new_true_high, new_true_low = self.__true_high_and_low(new_setup_ind-lookback-count, new_setup_ind)
        old_true_high, old_true_low = self.__true_high_and_low(old_setup_ind-lookback-count, new_setup_ind)
        new_size = new_true_high - new_true_low
        old_size = old_true_high - old_true_low
        return (old_size <= new_size) and (new_size <= 1.618*old_size)

    def __check_setup_trend(self, i, setupInd, side):
        count, lookback = self.setup_count, self.setup_lookback
        true_high, true_low = self.__true_high_and_low(setupInd-lookback-count, i)
        if side == "buy":
            return self.daily_data.iloc[i]["close"] > true_high
        if side == "sell":
            return self.daily_data.iloc[i]["close"] < true_low
        
    def generate_signal(self, i, t, side):
        self.obv.generate_signal(i, t)
        if t not in self.signal.index:  
            self.signal.loc[t] = 0
            
        if side == "buy":
            opposite_side, isSetup, setupInd = "sell", self.isSetupBuy, self.setupIndBuy
            countdown = self.countdownBuy
            countdown_period = self.buy_countdown_period
            univ_countdown_period = self.univ_buy_countdown_period
            oppo_univ_countdown_period = self.univ_sell_countdown_period
        elif side == "sell":
            opposite_side, isSetup, setupInd = "buy", self.isSetupSell, self.setupIndSell
            countdown = self.countdownSell
            countdown_period = self.sell_countdown_period
            univ_countdown_period = self.univ_sell_countdown_period
            oppo_univ_countdown_period = self.univ_buy_countdown_period

        if not isSetup:
            if self.check_setup(i, side): 
                self.setup(i, t, side, "S", univ_countdown_period)
        elif isSetup:
            if countdown < countdown_period:
                if self.__check_setup_trend(i, setupInd, side):
                    self.cancel_setup(t, side)
                    return 
                if self.check_setup(i, opposite_side):
                    self.cancel_setup(t, side)
                    self.setup(i, t, opposite_side, "S", oppo_univ_countdown_period)
                    return 
                if i != setupInd and self.check_setup(i, side) and self.__check_recycle(i, setupInd):
                    self.__recycle(i, t, side)
            self.__check_countdown(i, t, side)
    
    def plot_signal(self, ax1, ax2, ax3, start, end):
        price = self.daily_data.loc[start:end]["close"]
        signal = self.signal.loc[start:end]
        buy_countdown = signal[signal["buy countdown"]!=0]
        sell_countdown = signal[signal["sell countdown"]!=0]
        buy_setup = signal[signal["buy setup"]!=0]
        sell_setup = signal[signal["sell setup"]!=0]
        for t in buy_countdown.index:
            countdown = buy_countdown.loc[t,"buy countdown"]
            ax1.text(t, price.loc[t], countdown, color="darkgreen" if signal.loc[t,"buy"]!=0 else "seagreen",\
                     size="large" if signal.loc[t,"buy"]!=0 else "small", weight="bold")
        for t in sell_countdown.index:
            countdown = sell_countdown.loc[t,"sell countdown"]
            ax1.text(t, price.loc[t], countdown, color="darkred" if signal.loc[t,"sell"]!=0 else "red",\
                     size="large" if signal.loc[t,"sell"]!=0 else "small", weight="bold")
        for t in buy_setup.index:
            val = buy_setup.loc[t,"buy setup"]
            ax1.text(t, price.loc[t], val, color="green", size="medium", weight="bold", alpha=0.8)
        for t in sell_setup.index:
            val = sell_setup.loc[t,"sell setup"]
            ax1.text(t, price.loc[t], val, color="red", size="medium", weight="bold", alpha=0.8)
        ax2 = self.obv.plot_signal(ax2, start, end)
        return ax1, ax2, ax3

class EnhancedDeMark(DeMarkSequence):
    def __init__(self, hourly_data=None, 
                       setup_lookback=4, setup_count=9, 
                       buy_countdown_period=13, sell_countdown_period=13, 
                       buy_early_countdown_period=7, sell_early_countdown_period=9, 
                       notional=10000):
        super().__init__(hourly_data, setup_lookback, setup_count, buy_countdown_period, sell_countdown_period, notional)
        self.buy_early_countdown_period = buy_early_countdown_period
        self.sell_early_countdown_period = sell_early_countdown_period
        self.prevWeightedPrice = None
        self.bollingbands = BollingerBands(hourly_data, 
                                           alpha=0.2, 
                                           lookback_days=3, 
                                           k=3, j=1, big_CI=0.8, small_CI=0.75, 
                                           estimator="TSRV")
                                          
    def __recover_previous_setup(self, t, side):
        match = self.signal[(self.signal[f"{side} setup"]=="S") | (self.signal[f"{side} setup"]=="ES") | (self.signal[f"{side} setup"]=="R")]
        date = match.index[-1]
        self.signal.loc[t, side] = 0
        if side == "buy":
            self.countdownBuy = self.signal.loc[t, "buy countdown"]
            self.isSetupBuy = True
            self.setupIndBuy = self.signal.index.get_loc(date)
            
        elif side == "sell":
            self.countdownSell = self.signal.loc[t, "sell countdown"]
            self.isSetupSell = True
            self.setupIndSell = self.signal.index.get_loc(date)
    
    def __early_setup(self, i, t, side):
        oppositeSide = "sell" if side == "buy" else "buy"
        oppositeIsSetup = self.isSetupSell if side == "buy" else self.isSetupBuy
        early_countdown_period = self.buy_early_countdown_period if side == "buy" else self.sell_early_countdown_period
        self.setup(i, t, side, "ES", early_countdown_period)
        # New buy setup occurs, so cancel sell setup if there's any
        if oppositeIsSetup:
            self.cancel_setup(t, oppositeSide)
    
    def get_moments_quantile(self, t, q, mom):
        end, start = t-timedelta(days=1), t-timedelta(days=7)
        quantile = self.bollingbands.signal.loc[start:end, mom].quantile(q)
        return quantile
    
    def empirical_cdf(self, arr, x):
        arr.sort()
        n = len(arr)
        ind = np.searchsorted(arr, x)
        if ind == n: return 1
        if ind == 0: return 0
        if x in arr:
            return ind / n
        else:
            cdf_low, cdf_high = ind/n, (ind+1)/n
            fraction = (x-arr[ind-1]) / (arr[ind]-arr[ind-1])
            return cdf_low + (cdf_high-cdf_low)*fraction
        
    def moment_ecdf(self, t, x, mom):
        end, start = t-timedelta(days=1), t-timedelta(days=20)
        arr = np.sort(self.bollingbands.signal.loc[start:end, mom])
        return self.empirical_cdf(arr, x)
        
    def sizing_func(self, X, B, M):
        return 2 / (1+np.exp(-B*(X-M)))
    
    def get_weight_buying_price(self):
        signal = self.signal
        ind = signal[signal["sell"]!=0].last_valid_index()
        buySizes = signal.loc[ind:, "buy"]
        buySizes = buySizes[buySizes!=0]
        if not len(buySizes): return self.prevWeightedPrice
        prices = self.daily_data.loc[buySizes.index, "close"]
        quantities = (self.notional*buySizes/2) / prices
        weightedPrice = prices.dot(quantities)/quantities.sum()
        self.prevWeightedPrice = weightedPrice
        return weightedPrice

    def generate_signal(self, i, t, side):
        super().generate_signal(i, t, side)
        self.bollingbands.generate_signal(t)
        bollingbandsSignal = self.bollingbands.signal.loc[t]
        price = self.daily_data.loc[t,"close"]
        two_lag_price = self.daily_data.iloc[i-2]["close"]
        countdown_period = self.buy_countdown_period if side == "buy" else self.sell_countdown_period
        if side == "buy":
#             if (not self.isSetupBuy) and (demarkSignal["buy countdown"] < self.countdown_period) and (price < bollingbandsSignal["BigDown"]):
#                 self.__early_setup(i, t, "buy")
            if (not self.isSetupBuy) and (self.signal.loc[t, "buy countdown"] < countdown_period):
                qKurt, qSkew = self.get_moments_quantile(t, 0.5, "kurt"), self.get_moments_quantile(t, 0.3, "skew")
                kurt, skew = bollingbandsSignal["kurt"], bollingbandsSignal["skew"]
                if kurt <= qKurt and skew <= qSkew and self.check_setup(i, side, 3, 1): 
                    self.__early_setup(i, t, "buy")
            if self.signal.loc[t, "buy countdown"] >= countdown_period:
                if price > bollingbandsSignal["SmallDown"] and price < bollingbandsSignal["SmallUp"]:
                    self.__recover_previous_setup(t, "buy")
#                 elif price > bollingbandsSignal["BigUp"]:
#                     self.signal.loc[t, "buy"] = 0.5
#                 elif price < bollingbandsSignal["BigDown"]:
#                     self.signal.loc[t, "buy"] = 1.5
            if self.countdownBuy >= countdown_period-3:
                if price/two_lag_price-1 >= 0.05:
#                 if price >= self.daily_data.iloc[i-2]["high"]:
                    self.signal.loc[t, "buy countdown"] = self.countdownBuy
                    self.reset("buy")
                    self.signal.loc[t, "buy"] = 0.8
                    
            if self.signal.loc[t, "buy"] != 0:
                # mKurt, mSkew = self.get_moments_quantile(t, 0.5, "kurt"), self.get_moments_quantile(t, 0.5, "skew")
                kurt, skew = bollingbandsSignal["kurt"], bollingbandsSignal["skew"]
#                 if kurt <= mKurt and skew <= mSkew:
#                     self.signal.loc[t, "buy"] += 0.5
#                 elif kurt > mKurt and skew > mSkew:
#                     self.signal.loc[t, "buy"] -= 0.5
                if price < bollingbandsSignal["SmallDown"]:
                    size1 = self.sizing_func((price-bollingbandsSignal["SmallDown"])/bollingbandsSignal["SmallDown"], B=-7, M=0)
                elif price > bollingbandsSignal["SmallUp"]:
                    size1 = self.sizing_func((price-bollingbandsSignal["SmallUp"])/bollingbandsSignal["SmallUp"], B=-7, M=0)
                else:
                    size1 = 1
#                     size2 = self.sizing_func((self.ECDF(t, skew, "skew")+self.ECDF(t, kurt, "kurt"))/2, B=-15, M=0.5)
                size2 = self.sizing_func(self.moment_ecdf(t, skew, "skew"), B=-17, M=0.5)
                size3 = self.sizing_func(self.moment_ecdf(t, kurt, "kurt"), B=-17, M=0.5)
                self.signal.loc[t, "buy"] *= (size1+size2+size3)/3

        elif side == "sell":
#             if (not self.isSetupSell) and (demarkSignal["sell countdown"] < self.countdown_period) and (price > bollingbandsSignal["BigUp"]):
#                 self.__early_setup(i, t, "sell")
            if (not self.isSetupSell) and (self.signal.loc[t,"sell countdown"] < countdown_period):
                qKurt, qSkew = self.get_moments_quantile(t, 0.5, "kurt"), self.get_moments_quantile(t, 0.7, "skew")
                kurt, skew = bollingbandsSignal["kurt"], bollingbandsSignal["skew"]
                if kurt >= qKurt and skew >= qSkew and self.check_setup(i, side, 3, 1): 
                    self.__early_setup(i, t, "sell")
            if self.signal.loc[t,"sell countdown"] >= countdown_period:
                if price > bollingbandsSignal["SmallDown"] and price < bollingbandsSignal["SmallUp"]:
                    self.__recover_previous_setup(t, "sell")
#                 elif price < bollingbandsSignal["BigDown"]:
#                     self.signal.loc[t, "sell"] = 0.5
#                 elif price > bollingbandsSignal["BigUp"]:
#                     self.signal.loc[t, "sell"] = 1.5
            if self.countdownSell >= countdown_period-3:
                if price/two_lag_price-1 <= -0.05:
#                 if price < self.daily_data.iloc[i-2]["low"]:
                    self.signal.loc[t, "sell countdown"] = self.countdownSell
                    self.reset("sell")
                    self.signal.loc[t, "sell"] = 0.8
            
            if self.countdownSell == 1:
                weightedPrice = self.get_weight_buying_price()                                    
                if weightedPrice and price < weightedPrice: 
                    self.signal.loc[t, "sell countdown"] = 0
                    self.countdownSell = 0

            if self.signal.loc[t, "sell"] != 0:
                # mKurt, mSkew = self.get_moments_quantile(t, 0.5, "kurt"), self.get_moments_quantile(t, 0.5, "skew")
                kurt, skew = bollingbandsSignal["kurt"], bollingbandsSignal["skew"]
#                 if kurt >= mKurt and skew >= mSkew:
#                     self.signal.loc[t, "sell"] += 0.5
#                 elif kurt < mKurt and skew < mSkew:
#                     self.signal.loc[t, "sell"] -= 0.5
                if price < bollingbandsSignal["SmallDown"]:
                    size1 = self.sizing_func((price-bollingbandsSignal["SmallDown"])/bollingbandsSignal["SmallDown"], B=7, M=0)
                elif price > bollingbandsSignal["SmallUp"]:
                    size1 = self.sizing_func((price-bollingbandsSignal["SmallUp"])/bollingbandsSignal["SmallUp"], B=7, M=0)
                else:
                    size1 = 1
#                     size2 = self.sizing_func((self.ECDF(t, skew, "skew")+self.ECDF(t, kurt, "kurt"))/2, B=15, M=0.5)
                size2 = self.sizing_func(self.moment_ecdf(t, skew, "skew"), B=17, M=0.5)
                size3 = self.sizing_func(self.moment_ecdf(t, kurt, "kurt"), B=17, M=0.5)
                self.signal.loc[t, "sell"] *= (size1+size2+size3)/3
                
    def plot_signal(self, ax1, ax2, ax3, start, end):
        ax1, ax2, ax3 = super().plot_signal(ax1, ax2, ax3, start, end)
        ax1, ax3 = self.bollingbands.plot_signal(ax1, ax3, start, end)
        return ax1, ax2, ax3