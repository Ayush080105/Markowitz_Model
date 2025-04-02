# Portfolio Optimization using Monte Carlo Simulation

This Streamlit app allows users to perform portfolio optimization using Monte Carlo simulations. Users can input stock tickers and analyze optimal asset allocation based on historical data.

## Features
- Fetches historical stock price data from Yahoo Finance
- Calculates log daily returns for risk-return analysis
- Simulates **10,000** random portfolios
- Optimizes portfolio allocation using the **Sharpe ratio**
- Visualizes stock price trends and efficient frontier

## Requirements
- Python 3.8+
- Required Python libraries:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `yfinance`
  - `matplotlib`
  - `scipy`

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/portfolio-optimization.git
   cd portfolio-optimization
   ```

2. Install dependencies:
   ```sh
   pip install streamlit numpy pandas yfinance matplotlib scipy
   ```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Enter stock tickers (comma-separated) and specify a date range.

3. Click **Run Optimization** to:
   - Download historical stock data
   - Generate portfolios using Monte Carlo simulation
   - Optimize allocation for maximum Sharpe ratio
   - Visualize portfolio efficiency

## Code Explanation

### Portfolio Generation
- Uses **Monte Carlo simulation** to randomly allocate asset weights.
- Computes expected return and risk for each portfolio.

### Optimization
- Applies **Scipy’s SLSQP** algorithm to maximize the Sharpe ratio.

### Visualization
- Plots stock price trends and efficient frontier using Matplotlib.

## Contribution
Feel free to contribute by submitting issues or pull requests. If you encounter any problems, let us know!

## License
This project is licensed under the MIT License.

