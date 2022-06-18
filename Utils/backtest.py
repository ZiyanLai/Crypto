import sys
sys.path.append("/Users/ZiyanLai/Dropbox/Files/Crypto")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from tqdm import tqdm
import scipy.stats as st
from scipy.optimize import brute
# from yahoo_fin import stock_info as si
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from Utils.APIs import DataSourcer, Notion
from pytz import timezone
from copy import deepcopy
central = timezone("US/Central")

class Backtester:
    def __init__(self, Strategy, hourly_data=None, ticker=None, 
                       start=None, end=None, days=None, forceOptimize=None, benchmark=pd.DataFrame()):
        if not isinstance(hourly_data, pd.DataFrame) and not ticker:
            raise Exception("Need to provide a valid ticker, or provide hourly data.")
        if isinstance(hourly_data, pd.DataFrame):
            self.hourly_data = hourly_data.copy()
        else:
            print(f"Getting Live {ticker.upper()} Data...")
            self.hourly_data = DataSourcer(ticker).get_hourly_data()
        start = self.hourly_data.index.min() if not start else start
        end = self.hourly_data.index.max() if not end else end
        if days:
            start_time = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0) - timedelta(days=days)
            self.hourly_data = self.hourly_data[self.hourly_data.index >= start_time]
        else:
            self.hourly_data = self.hourly_data.loc[start:end]
        self.__Strategy = Strategy
        self.typicalPrice = self.hourly_data.resample("1d").apply(self.__get_trading_price)
        self.notional = 10000
        self.transactionCost = 0.0145 #percentage
        self.tqdm_disable = False
        self.preOptimizing = False
        self.optimizing = False
        self.optimizeLookbackPeriods = 60
        if len(benchmark):
            self.benchmark = benchmark
        else:
            # self.benchmark = si.get_data("DPI-USD")["adjclose"]
            dpi = DataSourcer("dpi").get_daily_data()
            self.benchmark = dpi.resample("1d").last()["prices"]

        self.jsonOptParams = pd.read_json("params/opt_params")
        self.base_start()
        if forceOptimize != False:
            if self.__to_optimize() or forceOptimize == True:
                newParams = self.optimize()
                self.jsonOptParams.loc[pd.to_datetime(datetime.today())] = newParams
                self.jsonOptParams.to_json("params/opt_params")

    def __to_optimize(self):
        today = datetime.today().replace(hour=0,minute=0,second=0,microsecond=0)
        tDay = today.day
        opt_params = pd.read_json("params/opt_params")
        recent_opt_datetime = opt_params.index.max().replace(hour=0,minute=0,second=0,microsecond=0)
        if tDay >= 15:
            target_opt_datetime = today.replace(day=15,hour=12,minute=0,second=0,microsecond=0)
        elif tDay < 15:
            target_opt_datetime = today.replace(day=1,hour=12,minute=0,second=0,microsecond=0)
        return recent_opt_datetime < target_opt_datetime

    def __update_params(self, params):
        setup_lookback, setup_count, \
        buy_early_countdown_period, sell_early_countdown_period, \
        buy_countdown_period, sell_countdown_period, \
        small_CI, big_CI, alpha = params
        self.strategy.setup_lookback = int(setup_lookback)
        self.strategy.setup_count = int(setup_count)
        self.strategy.univ_sell_countdown_period = int(sell_countdown_period)
        self.strategy.univ_buy_countdown_period = int(buy_countdown_period)
        self.strategy.sell_countdown_period = min(self.strategy.sell_countdown_period, int(sell_countdown_period))
        self.strategy.buy_countdown_period = min(self.strategy.buy_countdown_period, int(buy_countdown_period))
        self.strategy.buy_early_countdown_period = int(buy_early_countdown_period)
        self.strategy.sell_early_countdown_period = int(sell_early_countdown_period)
        self.strategy.bollingbands.big_CI = big_CI
        self.strategy.bollingbands.small_CI = small_CI
        self.strategy.bollingbands.big_up_z = st.norm.ppf(big_CI+(1-big_CI)/2)
        self.strategy.bollingbands.small_up_z = st.norm.ppf(small_CI+(1-small_CI)/2)
        self.strategy.bollingbands.big_down_z = st.norm.ppf((1-big_CI)/2)
        self.strategy.bollingbands.small_down_z = st.norm.ppf((1-small_CI)/2)
        self.strategy.bollingbands.alpha = alpha

    def __format_params(self):
        oldIndex = self.jsonOptParams.index
        newIndex = [(t+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0) for t in oldIndex]
        optParams = self.jsonOptParams.copy()
        optParams.index = newIndex
        optParams.loc[self.strategy.daily_data.index.min()] = optParams.iloc[0]
        optParams = optParams.sort_index(ascending=True)
        # fullOptParams.index = newIndex
        # for t in self.strategy.daily_data.index:
        #     if t not in newIndex:
        #         fullOptParams.loc[t] = np.nan
        # fullOptParams = fullOptParams.sort_index(ascending=True)
        # fullOptParams = fullOptParams.bfill().ffill()
        return optParams

    def base_start(self):
        self.strategy = self.__Strategy(hourly_data=self.hourly_data,
                                        notional=self.notional)
        self.capital = 0
        self.optParams = self.__format_params()
        self.tracker = pd.DataFrame(0, index=self.strategy.daily_data.index,
                                       columns=["quantity","pnl","cum pnl","price","buy","sell","stoploss"], 
                                       dtype=np.float64)
        
    def __get_trading_price(self, data):
        if "high" in data.columns and "low" in data.columns:
            tp = (data["close"]+data["high"]+data["low"])/3
        else:
            tp = data["close"]
        return tp.mean()
    
    def generate_signal(self):
        for i, t in enumerate(tqdm(self.tracker.index, desc="Generating Signal", disable=self.tqdm_disable)):
            if self.preOptimizing and i >= len(self.tracker)-self.optimizeLookbackPeriods:
                return
            if self.optimizing and i < len(self.tracker)-self.optimizeLookbackPeriods:
                continue
            if t in self.optParams.index:
                params = self.optParams.loc[t]
                self.__update_params(params)
            self.strategy.generate_signal(i,t,"buy")
            self.strategy.generate_signal(i,t,"sell")
    
    def __invalid_opt_condition(self, params):
        setup_lookback, setup_count, \
        buy_early_countdown_period, sell_early_countdown_period, \
        buy_countdown_period, sell_countdown_period, \
        small_CI, big_CI, alpha = params
        cond1 = buy_early_countdown_period != int(buy_early_countdown_period) \
                    or buy_countdown_period != int(buy_countdown_period)
        cond2 = sell_early_countdown_period != int(sell_early_countdown_period) \
                    or sell_countdown_period != int(sell_countdown_period)
        cond3 = setup_lookback != int(setup_lookback) or setup_count != int(setup_count)
        cond4 = buy_early_countdown_period > buy_countdown_period
        cond5 = sell_early_countdown_period > sell_countdown_period
        cond6 = big_CI <= small_CI
        return cond1 or cond2 or cond3 or cond4 or cond5 or cond6
    
    def __revert_to_previous_model(self, previousModel):
        self.tracker = previousModel.tracker
        self.strategy = previousModel.strategy        


    def __optimize_objective(self, params, *args):
        setup_lookback, setup_count, \
        buy_early_countdown_period, sell_early_countdown_period, \
        buy_countdown_period, sell_countdown_period, \
        small_CI, big_CI, alpha = params
        previousModel, = args
        if self.__invalid_opt_condition(params): return np.inf
        row = [int(setup_lookback), int(setup_count),
               int(buy_early_countdown_period), int(sell_early_countdown_period), 
               int(buy_countdown_period), int(sell_countdown_period), 
               small_CI, big_CI, alpha]

        self.fullOptParams.iloc[-self.optimizeLookbackPeriods:] = row
        self.__base_backtest()
        opt_pnl = self.tracker["cum pnl"][-1]
        opt_ret = opt_pnl / self.capital
        score = 0.5*opt_ret - 0.25*self.__downside_beta(self.tracker) + 0.25*self.__sortino_ratio(self.tracker)
        self.__revert_to_previous_model(previousModel)
        # print(f"buy_early_countdown_period={buy_early_countdown_period}, sell_early_countdown_period={sell_early_countdown_period}, buy_countdown_period={buy_countdown_period}, sell_countdown_period={sell_countdown_period}, small_CI={small_CI}, big_CI={big_CI}, alpha={alpha}, cum return={opt_ret}, score={score}")
        print(int(setup_lookback), int(setup_count), 
              int(buy_early_countdown_period), int(sell_early_countdown_period), 
              int(buy_countdown_period), int(sell_countdown_period), small_CI, big_CI, alpha, 
              "score:", score)
        return -score
    
    def optimize(self):
        self.tqdm_disable = True
        self.preOptimizing = True
        self.generate_signal()
        self.preOptimizing = False
        self.optimizing = True
        previousModel = deepcopy(self)
        print("Optimizing parameters...")
        ranges = (slice(3, 4.1, 1), slice(7, 9.1, 1),
                  slice(7, 10.1, 1), slice(7, 10.1, 1), 
                  slice(10, 13.1, 1), slice(10, 13.1, 1), 
                  slice(0.65, 0.86, 0.05), slice(0.8, 0.96, 0.05), slice(0.1, 0.31, 0.2))
        params = brute(self.__optimize_objective, ranges=ranges, args=(previousModel,))
        self.tqdm_disable = False
        self.optimizing = False
        return params
    
    def base_backtest(self):
        self.generate_signal()
        self.tracker.loc[self.tracker.index,["buy","sell"]] = self.strategy.signal.loc[self.tracker.index,["buy","sell"]]
        self.tracker.loc[self.tracker.index,"price"] = self.typicalPrice.loc[self.tracker.index]
        prev_price, quantity = 0, 0
        self.capital = 0
        for i, t in enumerate(self.tracker.index):
            price = self.tracker.loc[t,"price"]
            pnl = quantity * (price-prev_price)
            buyUnit = self.tracker.loc[t,"buy"]/2
            sellUnit = self.tracker.loc[t,"sell"]/2
            if buyUnit != 0:
                buyQuantity = buyUnit*self.notional / price
                self.capital += buyUnit*self.notional
                pnl -= buyQuantity*price*self.transactionCost
                quantity += buyQuantity
            if sellUnit != 0:
                sellQuantity = min(sellUnit*self.notional / price, quantity)
                pnl -= sellQuantity*price*self.transactionCost
                quantity -= sellQuantity
            prev_price = price
            self.tracker.loc[t,"quantity"] = quantity
            self.tracker.loc[t,"pnl"] = pnl
            
        self.tracker["cum pnl"] = self.tracker["pnl"].cumsum()
        self.tracker["return"] = self.tracker["pnl"] / self.notional

    def __max_drawdown(self, tracker):
        cum_pnl = tracker["cum pnl"]
        peak = bottom = max_dd = 0
        peakInd = bottomInd = 0
        for i in range(1, len(cum_pnl)-1):
            prev, curr, nxt = cum_pnl[i-1], cum_pnl[i], cum_pnl[i+1]
            if curr >= prev and curr >= nxt:
                peak = curr
                peakInd = i
            elif curr <= prev and curr <= nxt:
                bottom = curr
                bottomInd = i
            if peakInd < bottomInd:
                max_dd = max(max_dd, peak - bottom)
        return max_dd
    
    def __downside_beta(self, tracker):
        benchmark_ret = np.log(self.benchmark).diff(1)
        joint_ind = np.intersect1d(tracker.index, benchmark_ret.index)
        benchmark_ret = benchmark_ret.loc[joint_ind]
        ret = tracker["return"].loc[joint_ind]
        downside_benchmark_ret = benchmark_ret[benchmark_ret<0]
        downside_strategy_ret = ret.loc[downside_benchmark_ret.index]
        y, X = downside_strategy_ret, sm.add_constant(downside_benchmark_ret)
        model = sm.OLS(y, X).fit()
        return model.params.iat[1]
    
    def __sortino_ratio(self, tracker):
        # start_date, end_date = tracker.index.min(), tracker.index.max()
        # start_date, end_date = tracker.index.min()-timedelta(days=1), tracker.index.max()
        # sub_benchmark = self.benchmark.loc[start_date:end_date]
        benchmark_ret = np.log(self.benchmark).diff(1)
        joint_ind = np.intersect1d(tracker.index, benchmark_ret.index)
        benchmark_ret = benchmark_ret.loc[joint_ind]
        ret = tracker["return"].loc[joint_ind]
        sub_ret = ret.loc[(ret<benchmark_ret).index]
        sub_benchmark = benchmark_ret.loc[sub_ret.index]
        ratio = (ret-benchmark_ret).mean() / (np.square(sub_ret-sub_benchmark).mean())**0.5
        return ratio
    
    def plot_pnl(self, start=None, end=None, days=None):
        start = self.tracker.index.min() if not start else start
        end = self.tracker.index.max() if not end else end
        if days:
            start = end - timedelta(days=days)
        tracker = self.tracker.loc[start:end]
        price = tracker["price"]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35,15), gridspec_kw={'height_ratios': [1,1,1]})
        buysell = tracker[["buy","sell"]]
        buy_days = buysell[buysell["buy"]!=0].index
        sell_days = buysell[buysell["sell"]!=0].index
        stoploss_days = tracker[tracker["stoploss"]!=0].index
        ax1.plot(price, label="Price")
        ax1.set_ylabel("Price(USD)")
        for t in buy_days:
            b = round(buysell.loc[t,"buy"], 3)
            ax1.text(t, price.loc[t], b, color="green", size="large", weight="bold")
        for t in sell_days:
            s = round(buysell.loc[t,"sell"], 3)
            ax1.text(t, price.loc[t], s, color="red", size="large", weight="bold")
        for t in stoploss_days:
            s = round(tracker.loc[t,"stoploss"], 3)
            ax1.text(t, price.loc[t], f"${s}", color="black", size="medium", weight="bold")

        aggregate_tracker = tracker[["pnl","return"]].resample("1d").sum()
        aggregate_tracker["quantity"] = tracker["quantity"].resample("1d").last()
        aggregate_tracker["cum pnl"] = tracker["cum pnl"].resample("1d").last()

        ax2.bar(aggregate_tracker.index, aggregate_tracker["pnl"], label="Daily PnL", alpha=0.8)
        ax2.plot(aggregate_tracker["cum pnl"], color="darkorange", label="Cum PnL")
        ax4 = ax2.twinx()
        ax4.plot(aggregate_tracker["quantity"], color="crimson", label="Position")
        ax4.set_ylabel("Unit(Crypto)")
        ax4.grid(False)
        sns.histplot(aggregate_tracker["return"], kde=True, stat="probability", ax=ax3)
        ax3.set_xlabel("Return")
        ax1.set_title("Trades", fontsize=18)
        ax3.set_title("Return Distribution", fontsize=15)
        ax1.legend(loc=2)
        ax2.legend(loc=2)
        ax4.legend(loc=1)
        table_data = [[self.__max_drawdown(aggregate_tracker), self.__downside_beta(aggregate_tracker), self.__sortino_ratio(aggregate_tracker), \
                       self.tracker["cum pnl"].iat[-1], self.tracker["cum pnl"].iat[-1] / self.capital, \
                       aggregate_tracker["return"].mean(), aggregate_tracker["return"].std(), \
                       aggregate_tracker["return"].skew(), aggregate_tracker["return"].kurt()]]
        table = pd.DataFrame(table_data, columns=["Max Drawdown($)", "Downside Beta", "Sortino Ratio", \
                                                  "Cum PnL", "Strategy Return", \
                                                  "Return Mean", "Return Std", "Return Skew", "Return Kurtosis"], \
                             index=["stats"])
        plt.show()
        display(table)

    def plot_signal(self, start=None, end=None, days=None):
        start = self.tracker.index.min() if not start else start
        end = self.tracker.index.max() if not end else end
        if days:
            start = end - timedelta(days=days)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(35,15), gridspec_kw={'height_ratios': [2,1,1]})
        lines = ax1.plot(self.strategy.daily_data.loc[start:end]["close"], label="VWAP")
        ax1.set_ylabel("Price(USD)")
        ax1, ax2, ax3 = self.strategy.plot_signal(ax1, ax2, ax3, start, end)
        ax1.legend(loc=2)
        ax1.set_title("Signal", fontsize=18)
        plt.show()

class Trader(Backtester):
    def __init__(self, Strategy, hourly_data=None, ticker=None, start=None, end=None, days=365, forceOptimize=None):
        super().__init__(Strategy, hourly_data=hourly_data, 
                         ticker=ticker, start=start, end=end, days=days, forceOptimize=forceOptimize)
        self.notion = Notion()
        self.__start()

    def __start(self):
        self.base_start()
        self.closing_price = self.strategy.hourly_data["close"].resample("1d").last()
        index = [t.replace(hour=23) for t in self.closing_price.index[:-1]]
        index.append(self.strategy.hourly_data.index[-1])
        self.closing_price.index = index
        self.trades = self.notion.get_trading_history()
    
    def __display_today_signal(self):
        signal = self.strategy.signal
        table = pd.DataFrame(np.nan, columns=["Side", "Setup", "Countdown", "Curr Countdown", "Size"], index=["Status"])
        fill = False
        if self.strategy.isSetupBuy or signal.iloc[-1]["buy"]!=0:
            side = "buy"
            fill = True
        if self.strategy.isSetupSell or signal.iloc[-1]["sell"]!=0:
            side = "sell"
            fill = True
        if fill:
            countdown_period = int(self.strategy.buy_countdown_period) if side == "buy" else int(self.strategy.sell_countdown_period)
            countdown = int(self.strategy.countdownBuy) if side == "buy" else int(self.strategy.countdownSell)
            setup = signal[signal["buy setup"]!=0].iloc[-1]["buy setup"] if side == "buy" else signal[signal["sell setup"]!=0].iloc[-1]["sell setup"]
            table["Side"] = side
            table["Size"] = f"${self.notional * signal.iloc[-1][side]/2}"
            table["Setup"] = setup
            table["Countdown"] = countdown_period
            table["Curr Countdown"] = countdown
        display(table.style.set_caption("Today's signal"))
        
    def backtest(self):
        self.generate_signal()
        self.tracker.index = self.closing_price.index
        self.tracker["price"] = self.closing_price
        self.__display_today_signal()
        for t in self.trades.index:
            self.tracker.loc[t] = 0
            self.tracker.loc[t,"quantity"] = self.trades.loc[t,"quantity"]
            self.tracker.loc[t, "price"] = self.trades.loc[t,"price"]
        self.tracker = self.tracker.sort_index(ascending=True)

        prev_price, totalQuantity = 0, 0
        for i, t in enumerate(self.tracker.index):
            price = self.tracker.loc[t,"price"]
            pnl = totalQuantity * (price-prev_price)
            quantity = self.tracker.loc[t,"quantity"]
            if quantity != 0:
                self.capital += abs(quantity)*price
                totalQuantity += quantity        
                if quantity > 0:
                    self.tracker.loc[t,"buy"] = abs(quantity)
                elif quantity < 0:
                    self.tracker.loc[t,"sell"] = abs(quantity)
            if pnl <= -self.notional*0.10:
                self.tracker.loc[t,"stoploss"] = pnl
                
            prev_price = price
            self.tracker.loc[t,"quantity"] = totalQuantity
            self.tracker.loc[t,"pnl"] = pnl

        self.tracker["cum pnl"] = self.tracker["pnl"].cumsum()
        self.tracker["return"] = self.tracker["pnl"] / self.notional
        
    def plot_pnl(self):
        start = self.trades.index.min()
        super().plot_pnl(start=start)


if __name__ == '__main__':
    from Utils.strategy import EnhancedDeMark
    trader = Trader(EnhancedDeMark, ticker="eth", forceOptimize=False)
    trader.backtest()
