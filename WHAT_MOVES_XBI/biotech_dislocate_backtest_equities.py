import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import backtrader as bt
from backtrader.feeds import PandasData

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
        self.order = None  # Track pending orders

    def calculate_historical_volatility(self, data, window):
        returns = data.pct_change().dropna()
        rolling_volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_volatility.mean()

    def next(self):
        if self.order:  # Check if there is a pending order
            return

        ticker = self.datas[0]._name
        
        # Create a DataFrame from the data feed
        equity_df = pd.DataFrame({
            'Close': [self.datas[0].close[i] for i in range(-self.params.window, 0)]
        })

        if len(equity_df) < self.params.window:
            return  # Not enough data yet

        historical_vol = self.calculate_historical_volatility(equity_df['Close'], self.params.window)
        current_price = self.datas[0].close[0]

        # Assuming implied_volatility is derived or calculated elsewhere, or you can set a dummy value
        implied_volatility = historical_vol * self.params.threshold  # Example calculation

        high_threshold = historical_vol * self.params.threshold
        low_threshold = historical_vol * (1 - (self.params.threshold - 1))

        if implied_volatility > high_threshold:
            self.order = self.sell(data=self.datas[0], size=100)  # Sell signal

        elif implied_volatility < low_threshold:
            self.order = self.buy(data=self.datas[0], size=100)  # Buy signal

        if self.position:  # Close position if certain conditions are met
            if (self.position.size > 0 and implied_volatility > high_threshold) or \
               (self.position.size < 0 and implied_volatility < low_threshold):
                self.order = self.close(data=self.datas[0])  # Close the position

class SignalData(PandasData):
    cols = ['open', 'high', 'low', 'close', 'volume']
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
