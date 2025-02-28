import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, ljung_box

def detect_autocorrelation(returns):
    acf_val = acf(returns, nlags=1)[1]
    lb_stat, lb_pval, _ = ljung_box(returns, lags=1)
    if lb_pval < 0.05:
        if acf_val > 0:
            print("Significant positive autocorrelation detected. Suggest momentum strategy.")
        elif acf_val < 0:
            print("Significant negative autocorrelation detected. Suggest mean-reversion strategy.")
        else:
            print("Significant autocorrelation detected, but ACF is zero. No clear strategy.")
    else:
        print("No significant autocorrelation detected.")
    return acf_val, lb_pval

def get_position(returns, strategy='momentum', lookback=12, holding=12):
    T = len(returns)
    position = np.zeros(T)
    for t in range(lookback, T - holding + 1):
        cum_return = np.prod(1 + returns[t - lookback : t]) - 1
        if strategy == 'momentum':
            if cum_return > 0:
                position[t : t + holding] = 1
        elif strategy == 'mean_reversion':
            if cum_return < 0:
                position[t : t + holding] = 1
    return position

def calculate_strategy_return(returns, position):
    strategy_returns = returns * position
    total_return = np.prod(1 + strategy_returns) - 1
    return total_return

def calculate_buy_and_hold_return(returns):
    total_return = np.prod(1 + returns) - 1
    return total_return

# Example usage
np.random.seed(42)
returns = np.random.normal(0, 0.01, 100)
returns = pd.Series(returns)

acf_val, lb_pval = detect_autocorrelation(returns)
position = get_position(returns, strategy='momentum', lookback=12, holding=12)
strategy_return = calculate_strategy_return(returns, position)
buy_and_hold_return = calculate_buy_and_hold_return(returns)

print(f"Strategy return: {strategy_return:.2%}")
print(f"Buy-and-hold return: {buy_and_hold_return:.2%}")