import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Load the tickers from the CSV file
df_tickers = pd.read_csv("biotech_tickers.csv")
unique_tickers_list = sorted(df_tickers.Ticker.unique().tolist())

# Ensure 'XBI' is included in the list of biotech tickers
biotech_tickers = unique_tickers_list if 'XBI' in unique_tickers_list else unique_tickers_list + ['XBI']

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

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Implied Volatility')
    
    ax1.set_title(f'Volatility Surface for {ticker}')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Days to Expiration')
    ax1.set_zlabel('Implied Volatility')
    
    ax2 = fig.add_subplot(122)
    ax2.plot(equity_df.index, equity_df['Close'], label='Close Price', color='blue')
    ax2.set_title(f'Equity Price for {ticker}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

def suggest_trade(ticker, options_df, implied_volatility, historical_volatility, threshold):
    # Calculate the thresholds based on user input
    high_threshold = historical_volatility * threshold
    low_threshold = historical_volatility * (1 - (threshold - 1))  # Adjust to keep thresholds balanced

    if implied_volatility > high_threshold:
        trade_action = "Sell call options"
        strike_price = options_df.loc[options_df['impliedVolatility'].idxmax(), 'strike']
        trade_details = (f"Consider selling call options for {ticker} at a strike price of {strike_price:.2f} "
                         f"as implied volatility is high.")
    elif implied_volatility < low_threshold:
        trade_action = "Buy call options"
        strike_price = options_df.loc[options_df['impliedVolatility'].idxmin(), 'strike']
        trade_details = (f"Consider buying call options for {ticker} at a strike price of {strike_price:.2f} "
                         f"as implied volatility is low.")
    else:
        trade_action = "Hold, no significant trade opportunity."
        trade_details = "Current volatility is within normal range."

    return trade_action, trade_details

# Streamlit app
st.title("Biotech Dislocation Finder")

# Explanation Markdown
st.markdown("""
### How to Use This App
1. **Select a Ticker**: Choose a biotech stock from the dropdown.
2. **Select the Window**: Define the period (in days) over which to calculate historical volatility.
3. **Set the Threshold**: 
   - Enter a threshold value that helps identify when the implied volatility of options significantly diverges from the historical volatility of the stock.
   - **What is Implied Volatility?** It reflects the market's expectations of future volatility based on options pricing.
   - **What is Historical Volatility?** It measures how much the stock's price has fluctuated in the past.
   - **How Does the Threshold Work?** 
     - If the implied volatility is greater than the historical volatility multiplied by the threshold, this may suggest that options are overpriced (a market dislocation).
     - Conversely, if the implied volatility is significantly lower, options may be underpriced.
     - For example, with a threshold of 1.5, if the historical volatility is 20%, an implied volatility of over 30% (20% x 1.5) would indicate a potential opportunity to sell options, while an implied volatility below 15% (20% x 0.75) might suggest a buying opportunity.

The app calculates the implied volatility from the options data and compares it to the historical volatility of the equity. A significant difference indicates potential trading opportunities.
""")

selected_ticker = st.selectbox("Select a Ticker", biotech_tickers)
window_days = st.number_input("Select the window (in days) for historical volatility:", min_value=1, max_value=365, value=30)
threshold = st.number_input("Set the threshold for identifying dislocations:", min_value=1.0, value=1.5, step=0.1)

if st.button("Analyze"):
    try:
        options_df = get_options_data(selected_ticker)
        equity_df = get_equity_data(selected_ticker)

        if 'impliedVolatility' in options_df.columns:
            dislocated, implied_vol, historical_vol = identify_dislocations(options_df, equity_df, threshold)
            
            st.write(f"**Implied Volatility:** {implied_vol:.2f}, **Historical Volatility:** {historical_vol:.2f}")
            
            if dislocated:
                st.success(f"{selected_ticker} has a market dislocation.")
            else:
                st.warning(f"{selected_ticker} has no market dislocation.")
                
            # Suggest a trade based on the dislocation with the user-defined threshold
            trade_action, trade_details = suggest_trade(selected_ticker, options_df, implied_vol, historical_vol, threshold)
            st.write(f"**Trade Suggestion:** {trade_action}")
            st.write(f"**Details:** {trade_details}")
            
            plot_combined(selected_ticker, options_df, equity_df)
        else:
            st.error(f"No implied volatility data for {selected_ticker}.")
    
    except Exception as e:
        st.error(f"Error processing {selected_ticker}: {e}")

# Run the app using:
# streamlit run <script_name>.py
