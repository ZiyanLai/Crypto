import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ssl
import requests
import json
from pycoingecko import CoinGeckoAPI
from pytz import timezone
ssl._create_default_https_context = ssl._create_unverified_context
central = timezone("US/Central")

# Source data in Unix time
class DataSourcer:
    def __init__(self, ticker):
        self.ticker = ticker

    def format_data(self, df):
        # This time is in Unix
        index = pd.to_datetime(df["unix"], unit="s")
        df.index = index
        df = df.sort_index(ascending=True)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        return df

    def utc_to_central_time(self, time):
        is_dst = central.localize(time).dst().seconds != 0
        if is_dst:
            dt = timedelta(hours=-5)
        else:
            dt = timedelta(hours=-6)
        return time+dt

    # Reading from this URL will get data ending at 00:00:00 Unix of the current day (Yesterday's 6pm in CST)
    def get_new_data(self):
        old = pd.read_json(f"{self.ticker.lower()}_1h_data")
        old = self.format_data(old)
        if datetime.today().replace(hour=0,minute=0,second=0,microsecond=0) in old.index:
            return old
        new = pd.read_csv(f"https://www.cryptodatadownload.com/cdd/Bitstamp_{self.ticker.upper()}USD_1h.csv", skiprows=1)
        new = self.format_data(new)
        new = new[new.index > old.index.max()]
        data = pd.concat([old,new])
        data.to_json(f"{self.ticker.lower()}_1h_data")
        return data
    
    # This will get the live OHLC to fill up the above data gap
    def get_most_live_ohlc(self, url, params):
        r = requests.get(url, params=params)
        data = json.loads(r.text)
        if "bitstamp" in url:
            df = pd.DataFrame(data['data']['ohlc'], dtype=np.float64)
            df.index = [pd.to_datetime(int(t), unit="s") for t in df["timestamp"]]
            df = df.drop(columns="timestamp")
        elif "coinbase" in url:
            df = pd.DataFrame(data, dtype=np.float64).rename(columns={0:"time",1:"low",2:"high",3:"open",4:"close",5:"volume"}).set_index("time")
            df.index = [pd.to_datetime(int(t), unit="s") for t in df.index]
        df["Volume USD"] = df["volume"] * df["close"]
        df = df.drop(columns=["volume"])
        df = df.sort_index(ascending=True)
        return df
    
    # Concat the gap data with live data
    def get_hourly_data(self):
        raw_data = self.get_new_data()
        url = f"https://www.bitstamp.net/api/v2/ohlc/{self.ticker.lower()}usd/"
        params = {"end": int(pd.to_datetime(datetime.now()+timedelta(days=1)).timestamp()), "step":3600, "limit":200}
        live_data = self.get_most_live_ohlc(url, params)
        raw_data = raw_data.drop(columns=["unix","symbol",f"Volume {self.ticker.upper()}"])
        data = raw_data.sort_index(ascending=True)
        # The data in the most recent hour in "data" is not with complete 1-hour information
        # So replace this row with live data's complete 1-hour information
        live_data = live_data[live_data.index >= data.index.max()]
        replace_time = live_data.index.min()
        data = data[data.index != replace_time]
        # Then concat together
        hour_data = pd.concat([data, live_data])
        # Covert Unix time to CST
        hour_data.index = [self.utc_to_central_time(t) for t in hour_data.index]
        return hour_data

    def get_daily_data(self):
        cg = CoinGeckoAPI()
        ticker = self.ticker.lower()
        tickerMapping = {"eth":"ethereum", "btc":"bitcoin", "dpi":"defipulse-index"} 
        raw = cg.get_coin_market_chart_by_id(id=tickerMapping[ticker], vs_currency="usd", days="3000")
        data = pd.DataFrame(dtype=np.float64)
        for key, val in raw.items():
            time = pd.to_datetime([x[0] for x in val], unit="ms")
            time = [self.utc_to_central_time(t) for t in time]
            val = [x[1] for x in val]
            data[key] = val
        data.index = time
        return data

class Notion:
    def __init__(self, db_id="669cd3dc736048568be87afc0d20a129", api_key="secret_lH5xze8cDPonm9CCec1IuZj8Uc1tOcozWuhmAqlTwaa"):
        self.db_id = db_id
        self.api_key = api_key

    def get_trading_history(self):
        url = f"https://api.notion.com/v1/databases/{self.db_id}/query"
        headers = {
            "Accept": "application/json",
            "Notion-Version": "2021-08-16",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        trades = pd.DataFrame(columns=["ticker","price","quantity"])
        response = requests.post(url, headers=headers)
        results = json.loads(response.text)["results"]
        for row in results:
            info = row["properties"]
            date = pd.to_datetime(info["Timestamp"]["date"]["start"]).tz_localize(None)
            # date = pd.to_datetime(info["Timestamp"]["date"]["start"])
            ticker = info["Ticker"]["title"][0]["plain_text"]
            price = float(info["Exchange Rate($)"]["number"])
            quantity = float(info["Quantity"]["formula"]["number"])
            trades.loc[date] = [ticker, price, quantity]
        return trades
