import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Constants
START_DATE = "2023-01-01"
PORTFOLIO_FILE_PATH = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\ADAR1_Daily_Ticker_Data.xlsx"
FF5_FILE_PATH = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\F-F_Research_Data_5_Factors_2x3_daily.CSV"
INDEX_TICKERS = ['XBI', '^NBI', 'IWM']

# Load and preprocess portfolio data
def load_portfolio_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Trade Dt": "Date", "Group": "Ticker"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).ffill()
    df["Returns_Percent"] = df["P&L (%)"]
    return df[df['Date'] >= START_DATE]

# Load F-F 5 Factors data
def load_ff5_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    return df[df['Date'] >= START_DATE]

# Fetch historical adjusted close data
def fetch_biotech_index_data(tickers):
    data = yf.download(tickers, start=START_DATE)['Adj Close']
    daily_returns = data.pct_change().dropna()
    daily_returns.reset_index(inplace=True)
    daily_returns['Date'] = pd.to_datetime(daily_returns['Date'], utc=True).dt.tz_localize(None)
    return daily_returns[daily_returns['Date'] >= START_DATE]

# Merge data
def merge_data(portfolio, ff5, biotech_returns):
    portfolio['Date'] = portfolio['Date'].dt.tz_localize(None)
    ff5['Date'] = ff5['Date'].dt.tz_localize(None)
    biotech_returns['Date'] = pd.to_datetime(biotech_returns['Date']).dt.tz_localize(None)
    merged = pd.merge(portfolio, ff5, on='Date', how='left')
    final = pd.merge(merged, biotech_returns, on='Date', how='left')
    return final

# Calculate factor exposure
def calculate_factor_exposure(cleaned_df, date):
    filtered_df = cleaned_df[cleaned_df['Date'] == date]
    weights = filtered_df['Market Value (%)']
    factor_columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    exposure = (filtered_df[factor_columns].T * weights).T.sum()
    return exposure

# Load data
df_daily_portfolio = load_portfolio_data(PORTFOLIO_FILE_PATH)
df_ff5_daily = load_ff5_data(FF5_FILE_PATH)
biotech_index_daily_returns = fetch_biotech_index_data(INDEX_TICKERS)

# Execute the merging function
final_df = merge_data(df_daily_portfolio, df_ff5_daily, biotech_index_daily_returns)
df = final_df.dropna(subset=['Ticker'])

# Replace tickers as specified
df['Ticker'] = df['Ticker'].apply(lambda x: x[:-3] + '.T' if x.endswith(' JP') else x)
df['Ticker'] = df['Ticker'].replace('2746505Z DC', 'ASND')

# Streamlit app layout
st.title("ADAR1 Portfolio Exposure Calculator")

col1, col2 = st.columns([1, 2])

with col1:
    unique_dates = df['Date'].dt.date.unique()
    selected_date = st.selectbox("Select a date:", sorted(unique_dates))

with col2:
    if st.button("Calculate Factor Exposure"):
        cleaned_df = df.dropna()
        exposure = calculate_factor_exposure(cleaned_df, pd.to_datetime(selected_date))
        
        # Display results in a bar chart
        st.subheader(f"Factor Exposure on {selected_date}:")
        factor_names = ['Market', 'Size', 'Value', 'Profitability', 'Investment']
        exposure.index = factor_names  # Rename factors for clarity
        st.bar_chart(exposure)

# Step-by-step notes
st.markdown("""
### How This App Works:

1. **Load Data**: The app loads your portfolio data and the Fama-French 5 factors data.
2. **Data Preprocessing**: It cleans and prepares the data, ensuring all dates are aligned and unnecessary data is removed.
3. **Select Date**: You can choose a specific date from the dropdown menu.
4. **Calculate Exposure**: When you press the "Calculate Factor Exposure" button, the app computes the exposure of your portfolio to the selected factors.
5. **Display Results**: The factor exposures are displayed in a bar chart for easy visualization, showing how your portfolio is positioned against market factors.
""")

# Run the app
if __name__ == "__main__":
    st.title("Factor Exposure Calculator")
