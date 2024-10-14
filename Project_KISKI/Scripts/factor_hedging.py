import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import yfinance as yf
import getFamaFrenchFactors as gff
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load portfolio data
@st.cache_data
def load_portfolio_data(file_path):
    df = pd.read_excel(file_path)
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Returns'] = df['P&L (%)'].fillna(0)
    relevant_columns = ['AUM BOD', 'AUM EOD', 'P&L (%)', 'Returns']
    return df[relevant_columns]

# Calculate performance metrics
def calculate_performance_metrics(returns):
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 12
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    annualized_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_volatility
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(12)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return total_return, annualized_return, sharpe_ratio, sortino_ratio, max_drawdown

# Main function to run the Streamlit app
def main():
    st.title("Portfolio Analysis App")

    # Load portfolio data
    file_path = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\KISKI_Portfolio_Monthly.xlsx"
    portfolio_data = load_portfolio_data(file_path)

    # Get unique dates from the portfolio data
    unique_dates = portfolio_data.index.strftime('%Y-%m-%d').tolist()

    # Date range selection using dropdowns
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.selectbox("Start Date", unique_dates, index=0)
    with col2:
        end_date = st.selectbox("End Date", unique_dates, index=len(unique_dates)-1)

    # Convert selected dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter portfolio data based on selected date range
    r_a = portfolio_data.loc[start_date:end_date, 'Returns']

    # Get benchmark (XBI) data within the dynamic date range
    benchmark_symbol = "XBI"
    benchmark_prices = yf.download(benchmark_symbol, start=start_date, end=end_date)['Close']

    # Calculate benchmark returns (monthly)
    benchmark_returns = benchmark_prices.resample('M').last().pct_change().dropna()

    # Load Fama-French 5 Factor data
    ff5_monthly = gff.famaFrench5Factor(frequency='m')
    ff5_monthly['date_ff_factors'] = pd.to_datetime(ff5_monthly['date_ff_factors'])
    ff5_monthly.set_index('date_ff_factors', inplace=True)

    # Ensure all data is aligned
    returns_df = pd.DataFrame({
        'Portfolio': r_a,
        'Benchmark': benchmark_returns,
        'Mkt-RF': ff5_monthly['Mkt-RF'],
        'SMB': ff5_monthly['SMB'],
        'HML': ff5_monthly['HML'],
        'RMW': ff5_monthly['RMW'],
        'CMA': ff5_monthly['CMA'],
        'RF': ff5_monthly['RF']
    })
    returns_df = returns_df.dropna()

    # Extract aligned data
    r_a = returns_df['Portfolio']
    r_b = returns_df['Benchmark']
    ff_factors = returns_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    rf = returns_df['RF']

    # Calculate alpha and beta for Market-Neutral (XBI) portfolio
    X_market = sm.add_constant(r_b)
    model_market = sm.OLS(r_a - rf, X_market).fit()
    alpha_market, beta_market = model_market.params

    # Construct a Market-Neutral (XBI) portfolio
    portfolio_market_neutral = r_a - rf - beta_market * (r_b - rf)
    portfolio_market_neutral.name = "Market-Neutral (XBI) Portfolio"

    # Calculate alpha and betas for Fama-French 5-factor model
    X_ff5 = sm.add_constant(ff_factors)
    model_ff5 = sm.OLS(r_a - rf, X_ff5).fit()
    alpha_ff5, *betas_ff5 = model_ff5.params

    # Construct a Fama-French 5-factor neutral portfolio
    portfolio_ff5_neutral = r_a - rf - (betas_ff5[0] * ff_factors['Mkt-RF'] + 
                                        betas_ff5[1] * ff_factors['SMB'] + 
                                        betas_ff5[2] * ff_factors['HML'] + 
                                        betas_ff5[3] * ff_factors['RMW'] + 
                                        betas_ff5[4] * ff_factors['CMA'])
    portfolio_ff5_neutral.name = "FF5-Neutral Portfolio"

    # Calculate performance metrics for all portfolios
    metrics_original = calculate_performance_metrics(r_a)
    metrics_market_neutral = calculate_performance_metrics(portfolio_market_neutral)
    metrics_ff5_neutral = calculate_performance_metrics(portfolio_ff5_neutral)

    # Create a DataFrame for the comparison
    metrics_df = pd.DataFrame({
        'Metric': ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown'],
        'ADAR1 Portfolio': metrics_original,
        'Market-Neutral (XBI) Portfolio': metrics_market_neutral,
        'FF5-Neutral Portfolio': metrics_ff5_neutral
    })

    # Display the comparison
    st.subheader("Performance Metrics Comparison")
    st.dataframe(metrics_df.set_index('Metric').style.format("{:.4f}"))

    # Plot returns
    st.subheader("Portfolio Returns Comparison")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=r_a.index, y=r_a, name='ADAR1 Portfolio', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=portfolio_market_neutral.index, y=portfolio_market_neutral, name='Market-Neutral (XBI) Portfolio', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=portfolio_ff5_neutral.index, y=portfolio_ff5_neutral, name='FF5-Neutral Portfolio', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=r_b.index, y=r_b, name='XBI Returns', mode='lines+markers'), secondary_y=True)
    fig.update_layout(title="Returns of Original, Market-Neutral (XBI), FF5-Neutral Portfolios, and XBI",
                      xaxis_title="Date",
                      yaxis_title="Monthly Return",
                      legend_title="Portfolio")
    st.plotly_chart(fig)

    # Display factor exposures
    st.subheader("Factor Exposures (Betas)")
    st.text(model_ff5.summary().as_text())

    # Calculate and display R-squared and risk explanation
    r_squared_ff5 = model_ff5.rsquared
    risk_explained = r_squared_ff5 * 100
    st.subheader("Risk Analysis")
    st.write(f"R-squared for FF5 model: {r_squared_ff5:.4f}")
    st.write(f"Percentage of risk explained by FF5 factors: {risk_explained:.2f}%")
    st.write(f"Percentage of risk unexplained (idiosyncratic): {100 - risk_explained:.2f}%")

if __name__ == "__main__":
    main()