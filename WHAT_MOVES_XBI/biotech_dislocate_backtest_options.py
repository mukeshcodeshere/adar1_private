import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import backtrader as bt
from backtrader.feeds import PandasData
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from tscv import GapKFold
#import pyfolio as pf

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
pd.options.display.float_format = "{:.2f}".format
plt.style.use('ggplot')
sns.set(style='darkgrid', context='talk', palette='Dark2')


class ConvergenceStrategy(bt.Strategy):
    params = (
        ('threshold', 1.5),
        ('window', 30),
    )

    def __init__(self):
        self.data_close = self.datas[0].close

    def calculate_historical_volatility(self, data, window):
        returns = data.pct_change().dropna()
        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_volatility.mean()

    def get_options_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            options_data = []

            for expiration in expirations:
                opt_chain = stock.option_chain(expiration)
                calls = opt_chain.calls.assign(type='call', expiration=expiration)
                puts = opt_chain.puts.assign(type='put', expiration=expiration)
                options_data.extend([calls, puts])

            return pd.concat(options_data, ignore_index=True)
        except Exception as e:
            print(f"Error fetching options data for {ticker}: {e}")
            return pd.DataFrame()

    def identify_dislocations(self, options_df, equity_df):
        historical_volatility = self.calculate_historical_volatility(equity_df['Close'], self.params.window)
        options_df['impliedVolatility'] = pd.to_numeric(options_df['impliedVolatility'], errors='coerce')
        implied_volatility = options_df['impliedVolatility'].mean()

        dislocated = implied_volatility > historical_volatility * self.params.threshold
        return dislocated, implied_volatility, historical_volatility

    def next(self):
        ticker = self.datas[0]._name
        
        # Create a DataFrame from the data feed
        equity_df = pd.DataFrame({
            'Close': [self.datas[0].close[i] for i in range(-self.params.window, 0)]
        })

        if len(equity_df) < self.params.window:
            return  # Not enough data yet

        options_df = self.get_options_data(ticker)

        if options_df.empty:
            print(f"No options data found for {ticker}")
            return

        try:
            dislocated, implied_vol, historical_vol = self.identify_dislocations(options_df, equity_df)

            if dislocated:
                high_threshold = historical_vol * self.params.threshold
                low_threshold = historical_vol * (1 - (self.params.threshold - 1))

                if implied_vol > high_threshold:
                    max_idx = options_df['impliedVolatility'].idxmax()
                    if max_idx >= 0 and max_idx < len(options_df):
                        self.sell(data=self.datas[0], size=100)  # Example trade action

                elif implied_vol < low_threshold:
                    min_idx = options_df['impliedVolatility'].idxmin()
                    if min_idx >= 0 and min_idx < len(options_df):
                        self.buy(data=self.datas[0], size=100)  # Example trade action

        except Exception as e:
            print(f"Error processing {ticker}: {e}")



class SignalData(PandasData):
    cols = ['open', 'high', 'low', 'close', 'volume', 'predictedSignal']
    lines = tuple(cols)
    params = {c: -1 for c in cols}
    params.update({'Date': None})
    params = tuple(params.items())


def fetch_data():
    dataset = yf.download('^NBI', start='1994-01-01')
    return dataset.sort_index(ascending=True)


def run_backtest(prices_df, strategy_params):
    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.addstrategy(ConvergenceStrategy, **strategy_params)
    cerebro.adddata(SignalData(dataname=prices_df))
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.setcommission(commission=0.001)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print(f'Ending Portfolio Value: {final_value:.2f}')
    
    return final_value, results[0]


def main():
    dataset = fetch_data()
    
    # Define backtest parameters
    backtest_start = '2022-08-03'
    backtest_end = '2024-02-01'
    
    df_backtest = dataset[backtest_start:backtest_end]
    
    print(f'Backtest period: {backtest_start} to {backtest_end}')

    # Run the backtest
    strategy_params = {'threshold': 1.5, 'window': 30}
    final_value, result = run_backtest(df_backtest, strategy_params)

if __name__ == "__main__":
    main()
