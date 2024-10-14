import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import calendar
import getFamaFrenchFactors as gff

# Load and preprocess portfolio data
@st.cache_data
def load_portfolio_data(file_path):
    df_monthly_portfolio = pd.read_excel(file_path)
    df_monthly_portfolio.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df_monthly_portfolio.columns = df_monthly_portfolio.columns.str.strip()
    df_monthly_portfolio['Date'] = pd.to_datetime(df_monthly_portfolio['Date'])
    return df_monthly_portfolio

# Load Fama-French 5 Factor data
@st.cache_data
def load_ff5_data():
    df_ff5_monthly = gff.famaFrench5Factor(frequency='m')
    df_ff5_monthly['date_ff_factors'] = pd.to_datetime(df_ff5_monthly['date_ff_factors'])
    return df_ff5_monthly

# Merge portfolio and Fama-French data
def merge_data(df_monthly_portfolio, df_ff5_monthly):
    df_monthly = pd.merge(
        df_monthly_portfolio,
        df_ff5_monthly,
        left_on='Date',
        right_on='date_ff_factors',
        how='inner'
    )
    df_monthly['Monthly Return'] = df_monthly['P&L (%)']
    df_monthly.set_index('Date', inplace=True)
    return df_monthly

# Perform regression analysis and calculate factor contributions
def perform_analysis(df_monthly):
    X = sm.add_constant(df_monthly[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    y = df_monthly['Monthly Return']
    model = sm.OLS(y, X).fit()

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    for factor in factors:
        df_monthly[f'{factor}_Contribution'] = model.params[factor] * df_monthly[factor]
    df_monthly['Unexplained'] = y - model.predict(X)

    return df_monthly, factors

# Create interactive plot function
def create_interactive_plot(df_monthly, factors, selected_year, selected_month):
    selected_date = df_monthly[(df_monthly.index.year == selected_year) & 
                               (df_monthly.index.month == selected_month)].index

    if len(selected_date) == 0:
        st.warning(f"No data available for {calendar.month_name[selected_month]} {selected_year}")
        return None
    
    selected_date = selected_date[0]  # Get the first (and should be only) matching date
    data = df_monthly.loc[selected_date]
    
    contributions = [data[f'{factor}_Contribution'] for factor in factors]
    contributions.append(data['Unexplained'])
    
    labels = factors + ['Unexplained']
    total_return = data['Monthly Return']
    
    percentages = [100 * contrib / total_return for contrib in contributions]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=percentages, hole=.3)])
    fig.update_layout(
        title=f"Portfolio Return Breakdown for {calendar.month_name[selected_month]} {selected_year}",
        annotations=[dict(text=f'Total Return: {total_return:.2f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    return fig

def main():
    st.title("Portfolio Return Analysis")

    # File path for portfolio data
    file_path = r"C:\Users\MukeshwaranBaskaran\Downloads\Project_KISKI\Data\KISKI_Portfolio_Monthly.xlsx"

    # Load and process data
    df_monthly_portfolio = load_portfolio_data(file_path)
    df_ff5_monthly = load_ff5_data()
    df_monthly = merge_data(df_monthly_portfolio, df_ff5_monthly)

    # Perform analysis
    df_monthly, factors = perform_analysis(df_monthly)

    # Create sidebar for user input
    st.sidebar.header("Select Date")
    years = sorted(df_monthly.index.year.unique())
    months = range(1, 13)

    selected_year = st.sidebar.selectbox("Year", years)
    selected_month = st.sidebar.selectbox("Month", months, format_func=lambda x: calendar.month_name[x])

    # Create and show interactive plot
    fig = create_interactive_plot(df_monthly, factors, selected_year, selected_month)
    if fig:
        st.plotly_chart(fig)

    # Display detailed breakdown
    st.subheader("Detailed Breakdown")
    selected_data = df_monthly[(df_monthly.index.year == selected_year) & 
                               (df_monthly.index.month == selected_month)]

    if not selected_data.empty:
        data = selected_data.iloc[0]
        total_return = data['Monthly Return']
        st.write(f"Total Return: {total_return:.2f}%")
        for factor in factors + ['Unexplained']:
            contribution = data[f'{factor}_Contribution'] if factor != 'Unexplained' else data['Unexplained']
            percentage = 100 * contribution / total_return
            st.write(f"{factor}: {percentage:.2f}%")
    else:
        st.warning(f"No data available for {calendar.month_name[selected_month]} {selected_year}")

if __name__ == "__main__":
    main()
