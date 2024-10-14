import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import calendar
import getFamaFrenchFactors as gff

@st.cache_data
def load_portfolio_data(file_path):
    df_monthly_portfolio = pd.read_excel(file_path)
    df_monthly_portfolio.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df_monthly_portfolio.columns = df_monthly_portfolio.columns.str.strip()
    df_monthly_portfolio['Date'] = pd.to_datetime(df_monthly_portfolio['Date'])
    return df_monthly_portfolio

@st.cache_data
def load_ff5_data():
    df_ff5_monthly = gff.famaFrench5Factor(frequency='m')
    df_ff5_monthly['date_ff_factors'] = pd.to_datetime(df_ff5_monthly['date_ff_factors'])
    return df_ff5_monthly

def merge_data(df_monthly_portfolio, df_ff5_monthly):
    df_monthly = pd.merge(
        df_monthly_portfolio,
        df_ff5_monthly,
        left_on='Date',
        right_on='date_ff_factors',
        how='inner'
    )
    df_monthly.set_index('Date', inplace=True)
    return df_monthly

def calculate_alpha(df_monthly):
    df_monthly['Expected Return'] = df_monthly[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].sum(axis=1) + df_monthly['RF']
    df_monthly['Alpha'] = df_monthly['P&L (%)'] - df_monthly['Expected Return']
    return df_monthly

def create_interactive_plot(df_monthly, selected_year, selected_month):
    selected_date = df_monthly[(df_monthly.index.year == selected_year) & 
                               (df_monthly.index.month == selected_month)].index

    if len(selected_date) == 0:
        st.warning(f"No data available for {calendar.month_name[selected_month]} {selected_year}")
        return None, None, None
    
    selected_date = selected_date[0]
    data = df_monthly.loc[selected_date]
    
    factor_names = {
        'Mkt-RF': 'Market',
        'SMB': 'Size',
        'HML': 'Value',
        'RMW': 'Profitability',
        'CMA': 'Investment'
    }
    
    factors = list(factor_names.keys())
    performances = [data[factor] * 100 for factor in factors]
    
    fig = go.Figure(data=[go.Bar(x=[factor_names[factor] for factor in factors], y=performances)])
    fig.update_layout(
        title=f"Raw Factor Performance for {calendar.month_name[selected_month]} {selected_year}",
        xaxis_title="Fama-French 5 Factors",
        yaxis_title="Performance (%)"
    )
    
    return fig, data['P&L (%)'] * 100, data['Alpha'] * 100

def retrieve_data_for_correlation(df_monthly):
    relevant_data = df_monthly[['P&L (%)', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    return relevant_data

def calculate_correlation_matrix(dataframe):
    returns_columns = ['P&L (%)', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    correlation_matrix = dataframe[returns_columns].corr()
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix):
    factor_labels = {
        'P&L (%)': 'Portfolio Return',
        'Mkt-RF': 'Market',
        'SMB': 'Size',
        'HML': 'Value',
        'RMW': 'Profitability',
        'CMA': 'Investment'
    }

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=[factor_labels[col] for col in correlation_matrix.columns],
        y=[factor_labels[col] for col in correlation_matrix.columns],
        colorscale=[[0, 'blue'], [1, 'red']],
        colorbar=dict(title='Correlation', tickvals=[-1, 0, 1], ticktext=['-1 (Uncorrelated)', '0', '1 (Super Correlated)']),
    ))
    
    fig.update_layout(title='Correlation Matrix Heatmap', xaxis_title='Factors', yaxis_title='Factors')
    return fig

def calculate_sharpe_ratio(df_monthly, selected_year, selected_month):
    risk_free_rate = df_monthly['RF'].mean()
    portfolio_returns = df_monthly['P&L (%)']
    excess_returns = portfolio_returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(12)  # Annualized
    return sharpe_ratio

def calculate_sortino_ratio(df_monthly, selected_year, selected_month):
    risk_free_rate = df_monthly['RF'].mean()
    portfolio_returns = df_monthly['P&L (%)']
    excess_returns = portfolio_returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    sortino_ratio = (excess_returns.mean() / downside_returns.std()) * np.sqrt(12)  # Annualized
    return sortino_ratio

def calculate_active_premium(df_monthly, selected_year, selected_month):
    portfolio_return = df_monthly['P&L (%)'].mean() * 12  # Annualized
    benchmark_return = df_monthly['Mkt-RF'].mean() * 12 + df_monthly['RF'].mean() * 12  # Annualized
    active_premium = portfolio_return - benchmark_return
    return active_premium

def calculate_tracking_error(df_monthly, selected_year, selected_month):
    excess_returns = df_monthly['P&L (%)'] - (df_monthly['Mkt-RF'] + df_monthly['RF'])
    tracking_error = excess_returns.std() * np.sqrt(12)  # Annualized
    return tracking_error

def calculate_information_ratio(df_monthly, selected_year, selected_month):
    active_premium = calculate_active_premium(df_monthly, selected_year, selected_month)
    tracking_error = calculate_tracking_error(df_monthly, selected_year, selected_month)
    information_ratio = active_premium / tracking_error
    return information_ratio

def calculate_max_drawdown(df_monthly, selected_year, selected_month):
    cumulative_returns = (1 + df_monthly['P&L (%)']).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def main():
    st.title("Portfolio Return Analysis")

    file_path = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\KISKI_Portfolio_Monthly.xlsx"

    df_monthly_portfolio = load_portfolio_data(file_path)
    df_ff5_monthly = load_ff5_data()
    df_monthly = merge_data(df_monthly_portfolio, df_ff5_monthly)
    df_monthly = calculate_alpha(df_monthly)

    st.sidebar.header("Select Date")
    years = sorted(df_monthly.index.year.unique())
    months = range(1, 13)

    selected_year = st.sidebar.selectbox("Year", years)
    selected_month = st.sidebar.selectbox("Month", months, format_func=lambda x: calendar.month_name[x])

    fig, total_return, alpha = create_interactive_plot(df_monthly, selected_year, selected_month)
    if fig:
        st.plotly_chart(fig)

    st.subheader("Portfolio Performance")
    st.write(f"ADAR1 Portfolio Return for {calendar.month_name[selected_month]} {selected_year}: {total_return:.2f}%")
    st.write(f"Alpha Generated by ADAR1 for {calendar.month_name[selected_month]} {selected_year}: {alpha:.2f}%")

    # Calculate and display new performance metrics
    sharpe_ratio = calculate_sharpe_ratio(df_monthly, selected_year, selected_month)
    sortino_ratio = calculate_sortino_ratio(df_monthly, selected_year, selected_month)
    active_premium = calculate_active_premium(df_monthly, selected_year, selected_month)
    tracking_error = calculate_tracking_error(df_monthly, selected_year, selected_month)
    information_ratio = calculate_information_ratio(df_monthly, selected_year, selected_month)
    max_drawdown = calculate_max_drawdown(df_monthly, selected_year, selected_month)

    st.subheader("Performance Metrics")
    st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
    st.write(f"Active Premium: {active_premium:.2f}%")
    st.write(f"Tracking Error: {tracking_error:.2f}%")
    st.write(f"Information Ratio: {information_ratio:.2f}")
    st.write(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Calculate and display correlation matrix heatmap
    correlation_data = retrieve_data_for_correlation(df_monthly)
    correlation_matrix = calculate_correlation_matrix(correlation_data)
    st.subheader("Correlation")
    heatmap_fig = plot_correlation_heatmap(correlation_matrix)
    st.plotly_chart(heatmap_fig)

if __name__ == "__main__":
    main()