import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000

# Streamlit App
st.title("Portfolio Optimization using Monte Carlo Simulation")

# User Inputs
stocks = st.text_input("Enter stock tickers (comma-separated, e.g., AAPL, GOOG, MSFT):", "^BSESN").split(",")
start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2017-01-01"))


# Functions
def download_data(stocks, start_date, end_date):
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock.strip())
        stock_data[stock.strip()] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))
    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])


def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds,
                                 constraints=constraints)


def plot_data(data):
    fig, ax = plt.subplots()
    data.plot(figsize=(10, 5), ax=ax)
    st.pyplot(fig)


def plot_portfolios(portfolio_means, portfolio_risks, optimum, returns):
    fig, ax = plt.subplots()
    scatter = ax.scatter(portfolio_risks, portfolio_means, c=portfolio_means / portfolio_risks, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    optimal_stats = statistics(optimum['x'], returns)
    ax.scatter(optimal_stats[1], optimal_stats[0], color='red', marker='*', s=200, label='Optimal Portfolio')
    ax.set_title("Portfolio Optimization")
    ax.set_xlabel("Expected Volatility")
    ax.set_ylabel("Expected Return")
    ax.legend()
    st.pyplot(fig)


# Main Logic
if st.button("Run Optimization"):
    st.subheader("Downloading Data...")
    data = download_data(stocks, start_date, end_date)
    st.write("Data Overview:")
    st.write(data.tail())

    st.subheader("Stock Price Trends")
    plot_data(data)

    st.subheader("Calculating Returns...")
    log_daily_returns = calculate_return(data)

    st.subheader("Generating Portfolios...")
    weights, means, risks = generate_portfolios(log_daily_returns)

    st.subheader("Optimizing Portfolio...")
    optimum = optimize_portfolio(weights, log_daily_returns)
    st.write("Optimal Portfolio Weights:", optimum['x'].round(3))
    optimal_stats = statistics(optimum['x'], log_daily_returns)
    st.write(f"Expected Return in a year: {optimal_stats[0]:.2f}")
    st.write(f"Expected Volatility: {optimal_stats[1]:.2f}")
    st.write(f"Sharpe Ratio: {optimal_stats[2]:.2f}")

    st.subheader("Portfolio Visualization")
    plot_portfolios(means, risks, optimum, log_daily_returns)
    st.subheader("Percent of Money to Invest in according to your Portfolio: ")
    for stock, weight in zip(stocks, optimum['x']):
        st.write(f"The percentage of money to invest in {stock} is {weight * 100:.2f}%")
