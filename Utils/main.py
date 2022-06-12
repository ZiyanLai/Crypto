import sys
sys.path.append("/Users/ZiyanLai/Dropbox/Files/Crypto")
from Utils.strategy import EnhancedDeMark
from Utils.backtest import Trader
if __name__ == '__main__':
    trader = Trader(EnhancedDeMark, ticker="eth", forceOptimize=False)
    trader.backtest()