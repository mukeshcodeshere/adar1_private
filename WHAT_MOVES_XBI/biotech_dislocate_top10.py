import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

# Load the tickers from the CSV file
df_tickers = pd.read_csv("biotech_tickers.csv")
unique_tickers_list = sorted(df_tickers.Ticker.unique().tolist())
biotech_tickers = unique_tickers_list if 'XBI' in unique_tickers_list else unique_tickers_list + ['XBI']
biotech_tickers = unique_tickers_list if 'NBI' in unique_tickers_list else unique_tickers_list + ['NBI']

@st.cache_data
def get_options_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    options_data = []

    for expiration in expirations:
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls.assign(type='call', expiration=expiration)
        puts = opt_chain.puts.assign(type='put', expiration=expiration)
        options_data.extend([calls, puts])

    return pd.concat(options_data, ignore_index=True)

@st.cache_data
def get_equity_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1y")

def calculate_historical_volatility(equity_df, window=30):
    equity_df['returns'] = equity_df['Close'].pct_change()
    rolling_volatility = equity_df['returns'].rolling(window=window).std() * np.sqrt(252)
    return rolling_volatility.mean()

def identify_dislocations(options_df, equity_df, threshold=1.5):
    historical_volatility = calculate_historical_volatility(equity_df)
    options_df['impliedVolatility'] = pd.to_numeric(options_df['impliedVolatility'], errors='coerce')
    implied_volatility = options_df['impliedVolatility'].mean()

    dislocated = implied_volatility > historical_volatility * threshold
    return dislocated, implied_volatility, historical_volatility

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns.dropna(), (1 - confidence_level) * 100)

def predict_future_prices(equity_df, days=30):
    # Calculate returns
    equity_df['Returns'] = equity_df['Close'].pct_change()
    
    # Drop NaN values
    equity_df = equity_df.dropna(subset=['Returns'])
    
    # Prepare X and y
    X = np.array(range(len(equity_df))).reshape(-1, 1)
    y = equity_df['Returns'].values
    
    # Fit the model
    model = LinearRegression().fit(X, y)
    
    # Prepare future days
    future_days = np.array(range(len(equity_df), len(equity_df) + days)).reshape(-1, 1)
    
    # Predict future returns
    predicted_returns = model.predict(future_days)
    
    # Calculate predicted prices
    last_price = equity_df['Close'].iloc[-1]
    predicted_prices = last_price * (1 + predicted_returns).cumprod()
    
    return predicted_prices

def calculate_sharpe_ratio(equity_returns, risk_free_rate=0.01):
    excess_returns = equity_returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_options_greeks(option_data):
    option_data['d1'] = (np.log(option_data['stockPrice'] / option_data['strike']) +
                         (0.5 * option_data['sigma'] ** 2) * option_data['t']) / \
                        (option_data['sigma'] * np.sqrt(option_data['t']))
    
    option_data['Delta'] = norm.cdf(option_data['d1'])
    option_data['Gamma'] = norm.pdf(option_data['d1']) / (option_data['stockPrice'] * option_data['sigma'] * np.sqrt(option_data['t']))
    option_data['Vega'] = option_data['stockPrice'] * norm.pdf(option_data['d1']) * np.sqrt(option_data['t'])
    option_data['Theta'] = - (option_data['stockPrice'] * norm.pdf(option_data['d1']) * option_data['sigma']) / (2 * np.sqrt(option_data['t']))
    
    return option_data

def plot_combined(ticker, options_df, equity_df):
    options_df = options_df.dropna(subset=['strike', 'impliedVolatility'])
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    options_df['days_to_expiration'] = (options_df['expiration'] - pd.Timestamp.today()).dt.days

    strikes = np.sort(options_df['strike'].unique())
    days_to_expiration_unique = np.sort(options_df['days_to_expiration'].unique())
    
    X, Y = np.meshgrid(strikes, days_to_expiration_unique)
    Z = griddata(
        (options_df['strike'], options_df['days_to_expiration']),
        options_df['impliedVolatility'],
        (X, Y),
        method='cubic'
    )

    fig = plt.figure(figsize=(16, 12))

    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Implied Volatility')
    
    ax1.set_title(f'Volatility Surface for {ticker}', fontsize=16)
    ax1.set_xlabel('Strike Price', fontsize=14)
    ax1.set_ylabel('Days to Expiration', fontsize=14)
    ax1.set_zlabel('Implied Volatility', fontsize=14)
    ax1.view_init(elev=30, azim=210)  # Adjusting view angle for better visibility
    ax1.grid(True)  # Adding grid

    # Equity Price Plot
    ax2 = fig.add_subplot(122)
    ax2.plot(equity_df.index, equity_df['Close'], label='Close Price', color='blue', linewidth=2)
    
    # Plot predicted future prices
    predicted_prices = predict_future_prices(equity_df)
    future_dates = pd.date_range(start=equity_df.index[-1], periods=len(predicted_prices) + 1)[1:]
    ax2.plot(future_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--', linewidth=2)

    # Enhancements
    ax2.set_title(f'Equity Price for {ticker}', fontsize=16)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Price', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True)  # Adding grid for better readability

    plt.tight_layout()
    st.pyplot(fig)


def enhanced_trade_suggestions(ticker, options_df, equity_df, threshold):
    
     # Calculate historical returns for Sharpe Ratio
     equity_returns=equity_df["Close"].pct_change().dropna()
     sharpe_ratio=calculate_sharpe_ratio(equity_returns)

     # Get implied and historical volatility
     dislocated , implied_volatility , historical_volatility=identify_dislocations(options_df,equity_df , threshold)
     suggestion=""
     
     if implied_volatility>(historical_volatility*threshold):
         suggestion=f"Sell options for {ticker} - high implied volatility with Sharpe Ratio: {sharpe_ratio:.2f}."
     elif implied_volatility<(historical_volatility*(1-(threshold-1))):
         suggestion=f"Buy options for {ticker} - low implied volatility with Sharpe Ratio: {sharpe_ratio:.2f}."
     else:
         suggestion=f"Hold for {ticker} - volatility within normal range with Sharpe Ratio: {sharpe_ratio:.2f}."

     return suggestion

# Streamlit app
st.title("Biotech Dislocation Finder")

# Explanation Markdown
st.markdown("""
This application analyzes biotech stocks to identify market dislocations by comparing implied and historical volatilities.
It offers insights into potential trading opportunities based on volatility discrepancies.
""")

window_days=st.number_input("Select the window (in days) for historical volatility:",min_value=1,max_value=365,value=30)
threshold=st.number_input("Set the threshold for identifying dislocations:",min_value=1.0,value=1.5 , step=0.1)

def process_ticker(ticker):
    
     try:
         options_df=get_options_data(ticker)
         equity_df=get_equity_data(ticker)
         
         if "impliedVolatility" in options_df.columns:
             dislocated , implied_vol,historical_vol=identify_dislocations(options_df,equity_df , threshold)
             
             if dislocated:
                 return {
                     "Ticker":ticker,
                     "Implied Volatility":implied_vol,
                     "Historical Volatility":historical_vol,
                     "Dislocation Ratio":implied_vol/historical_vol}
             else:
                 enhanced_trade_suggestions(ticker , options_df , equity_df , threshold)
                 plot_combined(ticker , options_df , equity_df)
                 
         return None
    
     except Exception as e:
         st.error(f"Error processing {ticker}: {e}")
         
         return None

if st.button("Analyze All"):
    
  # Analyze all tickers including XBI and NBI separately for detailed insights.
  dislocated_stocks_xbi_nbi=[]
  dislocated_stocks=[]
  
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results_xbi_nbi=list(executor.map(process_ticker,['XBI','NBI']))
      results=list(executor.map(process_ticker,[x for x in biotech_tickers if x not in ['XBI','NBI']]))
      
      dislocated_stocks_xbi_nbi=[res for res in results_xbi_nbi if res is not None]
      dislocated_stocks=[res for res in results if res is not None]
      
  if dislocated_stocks_xbi_nbi:
      st.write("### XBI and NBI Analysis:")
      st.dataframe(pd.DataFrame(dislocated_stocks_xbi_nbi))
      
  if dislocated_stocks:
      dislocated_df=pd.DataFrame(dislocated_stocks)
      top_dislocated=dislocated_df.sort_values(by="Dislocation Ratio",ascending=False).head(10)

      st.write("### Top 10 Dislocated Stocks:")
      st.dataframe(top_dislocated)

      for stock in top_dislocated["Ticker"]:
          options_df=get_options_data(stock)
          equity_df=get_equity_data(stock)
          plot_combined(stock , options_df , equity_df)
          
  else:
      st.warning("No stocks with significant dislocations found.")