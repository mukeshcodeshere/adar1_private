# NEED GROUND TRUTH FOR COMPARISON 
# KISKI - https://tableau-server.kiski.com/#/site/_prod_adar1/views/PerformanceStatisticsADAR1/FactorContribution?:iid=1
# NOT ABLE TO COMPARE TO KISKI BECAUSE ITD FOR KISKI -> Inception to Date
# Bloomberg Portfolio Function to double confirm output?

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests
import zipfile
import io

# Streamlit app layout
st.set_page_config(page_title="ADAR1 Portfolio Exposure Calculator", layout="wide")
st.title("🎢 ADAR1 Portfolio Exposure Calculator")
st.markdown("### Analyze your portfolio's exposure to market factors.")

# Constants
START_DATE = "2023-01-01"
PORTFOLIO_FILE_PATH = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\ADAR1_Daily_Ticker_Data.xlsx"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
INDEX_TICKERS = ['XBI', '^NBI', 'IWM']

# Load and preprocess portfolio data
def load_portfolio_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    df.rename(columns={"Trade Dt": "Date", "Group": "Ticker"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).ffill()
    return df[df['Date'] >= START_DATE]

# Load F-F 5 Factors data with caching
@st.cache_data
def load_ff5_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                df = pd.read_csv(f, skiprows=3, sep=',', skipinitialspace=True)

        # Rename the first column to 'Date'
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # Drop any rows where 'Date' is not a valid date
        df = df[pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce').notna()]

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        return df[df['Date'] >= START_DATE]

    except Exception as e:
        st.error(f"An error occurred while loading F-F data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Fetch historical adjusted close data (cache it)
@st.cache_data
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

# Calculate factor exposure using weighted regression
def calculate_weighted_factor_exposure(cleaned_df, end_date, window=30):
    start_date = end_date - pd.Timedelta(days=window)
    filtered_df = cleaned_df[(cleaned_df['Date'] > start_date) & (cleaned_df['Date'] <= end_date)]
    
    if len(filtered_df) < 2:
        return None, None

    y = filtered_df['P&L (%)'] # RETURNS
    X = filtered_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    
    weights = np.abs(filtered_df['Market Value (%)'])
    
    adjusted_X = X.copy()
    for col in X.columns:
        adjusted_X[col] *= np.where(filtered_df['Market Value (%)'] < 0, -1, 1)

    model = LinearRegression().fit(adjusted_X, y, sample_weight=weights)  # Fit with absolute weights
    
    return model.intercept_, model.coef_

# Load data
df_daily_portfolio = load_portfolio_data(PORTFOLIO_FILE_PATH)
df_ff5_daily = load_ff5_data(FF5_URL)
biotech_index_daily_returns = fetch_biotech_index_data(INDEX_TICKERS)

# Execute the merging function
final_df = merge_data(df_daily_portfolio, df_ff5_daily, biotech_index_daily_returns)
df = final_df.dropna(subset=['Ticker'])

# Sidebar for user input
st.sidebar.header("User Input")
unique_dates = df['Date'].dt.date.unique()
selected_date = st.sidebar.selectbox("Select a date:", sorted(unique_dates))

if st.sidebar.button("Calculate Factor Exposure"):
    cleaned_df = df.dropna()
    selected_date = pd.to_datetime(selected_date)
    alpha, betas = calculate_weighted_factor_exposure(cleaned_df, selected_date)
    
    if betas is not None:
        st.subheader(f"Factor Exposures on {selected_date}:")
        factor_names = ['Market', 'Size', 'Value', 'Profitability', 'Investment']
        
        exposure_table = pd.DataFrame({'Factor': factor_names, 'Exposure': betas})
        st.table(exposure_table)

        # Plot bar chart
        st.subheader("Day-Level Factor Exposure Bar Chart")
        plt.figure(figsize=(10, 5))
        plt.bar(factor_names, betas, color=['blue', 'orange', 'green', 'red', 'purple'])
        plt.xlabel('Factors')
        plt.ylabel('Exposure')
        plt.title('Portfolio Factor Exposures')
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.error("Not enough data for the selected date range.")

steps = [
    {
        "title": "1. Load Portfolio Data",
        "description": (
            "Read the user's portfolio data from an Excel file. "
            "Clean the data by renaming columns and converting the date format. "
            "Filter the data to include only records from the specified start date."
        )
    },
    {
        "title": "2. Load F-F Factors Data",
        "description": (
            "Download the Fama-French 5 Factors data from a provided URL. "
            "Extract the zipped data and read it into a DataFrame. "
            "Clean and format the data, ensuring the date column is properly recognized and filtered from the start date."
        )
    },
    {
        "title": "3. Fetch Biotech Index Data",
        "description": (
            "Use yfinance to download historical adjusted close prices for specified biotech index tickers. "
            "Calculate daily returns based on these prices and format the data."
        )
    },
    {
        "title": "4. Merge Data",
        "description": (
            "Combine the portfolio data, F-F factors data, and biotech index returns into a single DataFrame based on the date. "
            "Ensure all date columns are in the same format to facilitate the merge."
        )
    },
    {
        "title": "5. Select Date for Analysis",
        "description": (
            "Allow the user to choose a specific date from the available dates in the merged DataFrame."
        )
    },
    {
        "title": "6. Calculate Factor Exposure",
        "description": (
            "Define a time window (e.g., 30 days) to filter the data leading up to the selected date. "
            "Extract the returns (dependent variable) and market factors (independent variables) for this filtered period. "
            "Calculate weights based on the portfolio's market value. "
            "Adjust the independent variables based on the sign of the market value. "
            "Fit a linear regression model using the adjusted market factors and returns, incorporating the calculated weights."
        )
    },
    {
        "title": "7. Present Results",
        "description": (
            "If valid factor exposures are calculated, display them in a table format. "
            "Generate a bar chart to visualize the portfolio’s exposure to each market factor for the selected date."
        )
    }
]

for step in steps:
    st.subheader(step["title"])
    st.write(step["description"])

# Run the app
if __name__ == "__main__":
    st.title("Factor Exposure Calculator")
