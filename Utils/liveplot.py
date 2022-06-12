import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.integrate import quad
import requests
import json
import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import timedelta
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from APIs import DataSourcer
import os
import requests
pio.templates.default = "plotly_dark"

class Params:
	k = 5
	j = 1
	lookback = 30	# in minutes
	plotHistory = 2 # in hours
	updateFreq = 3  # in seconds
	emaAlpha = 0.25
	upPercentile = 0.85
	side = None
	hardLimit = 3270
	reportTime = pd.to_datetime("today")-timedelta(days=1)
 
def TSRV(data, k, j):
    data = np.log(data)
    n = len(data)
    nbarK, nbarJ = (n-k+1)/k, (n-j+1)/j
    adj = (1-(nbarK/nbarJ))**(-1)
    RV_k = np.square(data - data.shift(k)).sum() / k
    RV_j = np.square(data - data.shift(j)).sum() / j
    RV = adj * (RV_k - (nbarK/nbarJ) * RV_j)
    sqrt = np.sqrt(max(0, RV))
    return sqrt

g = lambda x: min(x, 1-x)
g2 = lambda x: pow(g(x),2)
g3 = lambda x: pow(g(x),2)
g4 = lambda x: pow(g(x),4)
g2bar = quad(g2,0,1)[0]
g3bar = quad(g3,0,1)[0]
g4bar = quad(g4,0,1)[0]

def f(grid, k):
	res = 0
	ret = grid.diff(1)
	for i in range(1,len(ret)):
		res += g(i/k)*ret[i]
	return res

def f_bar(grid, k):
	res = 0
	ret = grid.diff(1)
	for i in range(1,len(ret)):
		res += ((g(i/k)-g((i-1)/k))*ret[i])**2
	return res

def realized_kurt(data, k):
	data = np.log(data)
	deltaY = data.rolling(window=k).apply(f, args=(k,)).dropna()
	deltaYBar = data.rolling(window=k).apply(f_bar, args=(k,)).dropna()
	realizedKurt = (deltaY**4).sum() / (k*g4bar)
	realizedVar = ((deltaY**2).sum()/k - deltaYBar.sum()/(2*k))/g2bar
	return realizedKurt/(realizedVar**2)

def realized_skew(data, k):
	data = np.log(data)
	deltaY = data.rolling(window=k).apply(f, args=(k,)).dropna()
	deltaYBar = data.rolling(window=k).apply(f_bar, args=(k,)).dropna()
	realizedSkew = (deltaY**3).sum() / (k*g3bar)
	realizedVar = ((deltaY**2).sum()/k - deltaYBar.sum()/(2*k))/g2bar
	return realizedSkew/(realizedVar**(3/2))

def OBV(data):
	price = data["close"]
	volume = data["Volume USD"]
	obv = pd.DataFrame(columns=["OBV"], dtype="float64")
	for i, t in enumerate(data.index):
		if i == 0:
			obv.loc[t,"OBV"] = volume.iat[0]
			continue
		price_today = price.iat[i]
		price_yesterday = price.iat[i-1]
		obv_yesterday = obv.iloc[i-1]["OBV"]
		if price_today > price_yesterday:
			obv.loc[t,"OBV"] = obv_yesterday + volume.iat[i]
		elif price_today < price_yesterday:
			obv.loc[t,"OBV"] = obv_yesterday - volume.iat[i]
		else:
			obv.loc[t,"OBV"] = obv_yesterday
		obv["delta OBV"] = obv["OBV"].diff(1)
	return obv

def get_live_data(ticker):
	url = f"https://api.exchange.coinbase.com/products/{ticker}-USD/candles"
	params={"granularity":60}
	dataSourcer = DataSourcer(ticker)
	data = dataSourcer.get_most_live_ohlc(url, params)
	data.index = [dataSourcer.utc_to_central_time(t) for t in data.index]
	return data

app = dash.Dash(__name__)

app.layout = html.Div(
	[
		html.Div(id='live-update-text'),
		dcc.Graph(id='live-update-ohlc'),
		dcc.Graph(id='live-update-indicators'),
		dcc.Interval(
			id='interval-component',
			interval = Params.updateFreq*1000,
			n_intervals = 0)
	]
)

@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_prices(n):
	data = get_live_data("ETH")
	latest = data.loc[data.index.max()]
	price = latest["close"]
	volumeUSD = round(latest["Volume USD"],2)
	r = requests.get("https://api.coinbase.com/v2/prices/ETH-USD/buy")
	ask = float(json.loads(r.text)["data"]["amount"])
	r = requests.get("https://api.coinbase.com/v2/prices/ETH-USD/sell")
	bid = float(json.loads(r.text)["data"]["amount"])
	style = {'padding':'5', 'fontSize':'25px'}
	return [
		html.Center(f"Price: ${price:,} - Volume: ${volumeUSD:,}", style=style),
		html.Center(
			[
				html.Span(f"Bid: ${bid:,}", style={'fontSize':'25px', 'color':'Red'}),
				html.Span(" - ", style={'fontSize':'25px'}),
				html.Span(f"Ask: ${ask:,}", style={'fontSize':'25px', 'color':'Green'})
			]
		),
	]

@app.callback(
	Output('live-update-indicators', 'figure'),
	[Input('interval-component', "n_intervals")]
)
def update_graph_indicators(n):
	data = get_live_data("ETH")	
	start = data.index[-1]-timedelta(hours=Params.plotHistory)
	end = data.index[-1]
	obv = OBV(data)
	kurt = data["close"].rolling(window=Params.lookback).apply(realized_kurt,args=(Params.k,))
	skew = data["close"].rolling(window=Params.lookback).apply(realized_skew,args=(Params.k,))
	displayOBV = obv.loc[start:end]
	displayKurt = kurt.loc[start:end]
	displaySkew = skew.loc[start:end]
	fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}],[{"secondary_y": False}]])
	fig.add_trace(go.Scatter(x=displayOBV.index, y=displayOBV["OBV"], line=dict(color="crimson",dash="dashdot",width=2), name="OBV"),
				  row=1,col=1,secondary_y=False)
	fig.add_trace(go.Bar(x=displayOBV.index, y=displayOBV["delta OBV"], opacity=0.5, marker=dict(color="darkcyan"), name="delta OBV"),
				  row=1, col=1,secondary_y=True)
	fig.add_trace(go.Scatter(x=displayKurt.index, y=displayKurt, line=dict(color="crimson",width=2), 
							 name=f"{Params.lookback}-min Kurt"), row=2,col=1)
	fig.add_trace(go.Scatter(x=displaySkew.index, y=displaySkew, line=dict(color="darkcyan",width=2), 
							 name=f"{Params.lookback}-min Skew"), row=2,col=1)
	fig['layout']['yaxis2']['showgrid'] = False
	fig['layout']['yaxis2']['zerolinewidth'] = 3
	fig.update_layout(
		autosize=False,
		width=1900,
		height=550,
		legend=dict(
			yanchor="top",
			y=1.04,
			xanchor="right",
			x=1.04
		),
		hovermode="x unified"
	)
	fig.update_xaxes(
		dtick=600000,
	)
	now = pd.to_datetime("today")
	currPrice = data["close"].iat[-1]
	buyQ = data["close"].quantile(0.2)
	sellQ = data["close"].quantile(0.8)
	high = data["close"].max()
	low = data["close"].min()
	if Params.side == "buy" and kurt.iat[-1] <= kurt.quantile(0.5) and skew.iat[-1] <= skew.quantile(0.3) \
							and currPrice <= buyQ and currPrice <= Params.hardLimit:
		if now-Params.reportTime >= timedelta(minutes=10):
			os.system("say " + "buy")
			requests.post('https://api.mynotifier.app', {
						  "apiKey": "7345969f-db91-41e2-9191-ca329b6bd9a2",
						  "message": "Buy Alert",
						  "description": f"Spot: {currPrice}\n5hr 0.2Q: {buyQ}\n5hr high: {high}\n5hr low: {low}\n",
						  "type": "info", # info, error, warning or success
						})
			Params.reportTime = now
	if Params.side == "sell" and kurt.iat[-1] >= kurt.quantile(0.5) and skew.iat[-1] >= skew.quantile(0.7) \
							 and currPrice >= sellQ and currPrice >= Params.hardLimit:
		if now-Params.reportTime >= timedelta(minutes=10):
			os.system("say " + "sell")
			requests.post('https://api.mynotifier.app', {
						  "apiKey": "7345969f-db91-41e2-9191-ca329b6bd9a2",
						  "message": "Sell Alert",
						  "description": f"Spot: {currPrice}\n5hr 0.8Q: {sellQ}\n5hr high: {high}\n5hr low: {low}\n",
						  "type": "info", # info, error, warning or success
						})
			Params.reportTime = now
	return fig

@app.callback(
	Output('live-update-ohlc', 'figure'),
	[Input('interval-component', 'n_intervals')] 
)
def update_graph_ohlc(n):
	data = get_live_data("ETH")
	ema = data['close'].ewm(alpha=Params.emaAlpha).mean()
	tsrv = data["close"].rolling(window=Params.lookback).apply(TSRV, args=(Params.k,Params.j))/np.sqrt(Params.lookback)
	up = (1+st.norm.ppf(Params.upPercentile)*tsrv)*ema
	down = (1+st.norm.ppf(1-Params.upPercentile)*tsrv)*ema
	data["ema"] = ema
	data["tsrv"] = tsrv
	data["up"] = up
	data["down"] = down
	start = data.index[-1]-timedelta(hours=Params.plotHistory)
	end = data.index[-1]
	displayData = data.loc[start:end]
	# fig = go.Figure(data=[go.Candlestick(x=displayData.index, open=displayData["open"], low=displayData["low"], high=displayData["high"], close=displayData["close"], name="OHLC"),
	# 					  go.Scatter(x=displayData.index, y=displayData["ema"], line=dict(color="lavender",dash="dash",width=1), name="EMA"),
	# 					  go.Scatter(x=displayData.index, y=displayData["up"], 
	# 								line=dict(color="gold",dash="dashdot",width=1.5), 
	# 								name="{:.0%} C.I.".format(Params.upPercentile)),
	# 					  go.Scatter(x=displayData.index, y=displayData["down"], 
	# 								line=dict(color="gold",dash="dashdot",width=1.5), 
	# 								name="{:.0%} C.I.".format(1-Params.upPercentile))
	# 					])
	fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
	fig.add_trace(go.Candlestick(x=displayData.index, open=displayData["open"], 
								low=displayData["low"], high=displayData["high"], 
								close=displayData["close"], name="OHLC"))
	fig.add_trace(go.Scatter(x=displayData.index, y=displayData["ema"], 
							line=dict(color="lavender",dash="dash",width=1), name="EMA"))
	fig.add_trace(go.Scatter(x=displayData.index, y=displayData["up"], 
							line=dict(color="gold",dash="dashdot",width=1.5), 
							name="{:.0%} C.I.".format(Params.upPercentile)))
	fig.add_trace(go.Scatter(x=displayData.index, y=displayData["down"], 
							line=dict(color="gold",dash="dashdot",width=1.5), 
							name="{:.0%} C.I.".format(1-Params.upPercentile)))
	fig.update_layout(
		autosize=False,
		width=1900,
		height=650,
		legend=dict(
			yanchor="top",
			y=1.04,
			xanchor="right",
			x=1.04
		),
		hovermode="x"
	)
	fig.update_xaxes(
		dtick=600000,
	)
	return fig

if __name__ == '__main__':
	app.run_server(debug=True)
