
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------
# 1. Daily Log Returns
# ------------------------------------------------------------

def compute_log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Compute daily log returns using efficient NumPy array operations.
    ln(P_t / P_{t-1})
    """
    if isinstance(prices, pd.DataFrame):
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return log_ret
    else:
        log_ret = np.log(prices / prices.shift(1)).dropna()
        return log_ret


def compute_daily_pct_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Simple daily percentage returns: (P_t - P_{t-1}) / P_{t-1}
    Used for display in the returns chart.
    """
    return prices.pct_change().dropna()


# ------------------------------------------------------------
# 2. Value at Risk (VaR)
# ------------------------------------------------------------

def compute_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss not exceeded with a given confidence level.
    Returns a positive number representing the loss amount.
    e.g., VaR 95% = -5th percentile of returns
    """
    if len(returns) == 0:
        return 0.0
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    return float(-cutoff)


def compute_var_table(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VaR at 95% and 99% for each ticker in returns DataFrame.

    Returns:
        DataFrame with columns ['Ticker', 'VaR_95%', 'VaR_99%']
    """
    rows = []
    for ticker in returns.columns:
        col = returns[ticker].dropna()
        rows.append({
            "Ticker": ticker,
            "VaR_95%": round(compute_var(col, 0.95) * 100, 3),
            "VaR_99%": round(compute_var(col, 0.99) * 100, 3),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 3. Monte Carlo Simulation (Minimum 10,000 Runs)
# ------------------------------------------------------------

def run_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 10_000,
    n_days: int = 252,
    initial_value: float = 100.0,
) -> np.ndarray:
    """
    Stochastic Monte Carlo simulation using Geometric Brownian Motion.
    Forecasts portfolio value distribution 1 year (252 trading days) into the future.

    Args:
        returns: Historical daily log returns for one asset/portfolio.
        n_simulations: Number of simulation paths (≥10,000 per spec).
        n_days: Trading days to project forward (252 = 1 year).
        initial_value: Starting portfolio value (default 100 for % representation).

    Returns:
        np.ndarray of shape (n_simulations,) — final portfolio values after n_days.
    """
    mu = float(returns.mean())
    sigma = float(returns.std())

    # Vectorised GBM: shape (n_days, n_simulations)
    dt = 1.0
    random_shocks = np.random.standard_normal((n_days, n_simulations))
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks)

    # Cumulative product along time axis → portfolio paths
    price_paths = initial_value * np.cumprod(daily_returns, axis=0)   # (n_days, n_simulations)

    # Return only the terminal values for the distribution histogram
    return price_paths[-1]


def run_monte_carlo_paths(
    returns: pd.Series,
    n_simulations: int = 200,
    n_days: int = 252,
    initial_value: float = 100.0,
) -> np.ndarray:
    """
    Return full price paths (for path chart) — limited to 200 paths for performance.

    Returns:
        np.ndarray of shape (n_days, n_simulations)
    """
    mu = float(returns.mean())
    sigma = float(returns.std())
    random_shocks = np.random.standard_normal((n_days, n_simulations))
    daily_returns = np.exp((mu - 0.5 * sigma ** 2) + sigma * random_shocks)
    return initial_value * np.cumprod(daily_returns, axis=0)


def monte_carlo_summary(terminal_values: np.ndarray) -> dict:
    """
    Compute descriptive statistics on MC terminal distribution.
    Validates statistical shape (skewness, kurtosis) against historical behavior.
    """
    return {
        "mean": float(np.mean(terminal_values)),
        "median": float(np.median(terminal_values)),
        "std": float(np.std(terminal_values)),
        "skewness": float(stats.skew(terminal_values)),
        "kurtosis": float(stats.kurtosis(terminal_values)),
        "var_95": float(np.percentile(terminal_values, 5)),
        "var_99": float(np.percentile(terminal_values, 1)),
        "p10": float(np.percentile(terminal_values, 10)),
        "p90": float(np.percentile(terminal_values, 90)),
    }


# ------------------------------------------------------------
# 4. Correlation Matrix
# ------------------------------------------------------------

def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Pearson correlation matrix using NumPy matrix multiplication
    for high-speed computation (essential for Portfolio Variance calculations).

    Returns:
        DataFrame with ticker labels for use in heatmap visualizations.
    """
    clean = returns.dropna()
    arr = clean.values.T   # shape (n_assets, n_observations)
    n = arr.shape[1]

    # Demean
    arr_mean = arr - arr.mean(axis=1, keepdims=True)

    # Covariance via matrix multiply
    cov = (arr_mean @ arr_mean.T) / (n - 1)

    # Normalise to correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)   # fix floating-point edge cases

    return pd.DataFrame(corr, index=clean.columns, columns=clean.columns)


# ------------------------------------------------------------
# 5. Rolling Volatility (30-day)
# ------------------------------------------------------------

def compute_rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 30,
    annualise: bool = True,
) -> pd.Series | pd.DataFrame:
    """
    30-day rolling standard deviation of returns — 
    a key indicator of market uncertainty.

    Args:
        annualise: If True, multiply by sqrt(252) to express as annualised vol.
    """
    roll_std = returns.rolling(window=window).std()
    if annualise:
        roll_std = roll_std * np.sqrt(252)
    return roll_std


# ------------------------------------------------------------
# 6. Portfolio Variance
# ------------------------------------------------------------

def compute_portfolio_variance(
    returns: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> float:
    """
    Portfolio variance using NumPy matrix multiplication:
        σ²_p = w^T · Σ · w
    where Σ is the covariance matrix and w is the weight vector.
    """
    clean = returns.dropna()
    n_assets = clean.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets   # equal weight

    # Annualised covariance matrix
    cov_matrix = clean.cov() * 252
    port_var = weights @ cov_matrix.values @ weights
    return float(port_var)


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(42)
    dummy_prices = pd.DataFrame(
        np.cumprod(1 + np.random.normal(0.0005, 0.015, (252, 3)), axis=0) * 100,
        columns=["A", "B", "C"],
    )
    rets = compute_log_returns(dummy_prices)
    print("Log returns shape:", rets.shape)

    corr = compute_correlation_matrix(rets)
    print("Correlation matrix:\n", corr.round(3))

    mc_vals = run_monte_carlo(rets["A"])
    summary = monte_carlo_summary(mc_vals)
    print("Monte Carlo summary:", summary)

    var_table = compute_var_table(rets)
    print("VaR table:\n", var_table)
