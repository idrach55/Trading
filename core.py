import pandas as pd
import numpy as np
import scipy.stats as stat

"""
Given pd.Series S_t, compute the rolling annualized n-day realized volaility
"""
def rolling_realized_vol(S_t,n):
    return np.sqrt((np.log(S_t/S_t.shift(1))**2).rolling(n).sum()*252/n).iloc[n:]

"""
Given pd.Series S_t, compute the annualized realized vol
"""
def realized_vol(S_t):
    return np.sqrt((np.log(S_t/S_t.shift(1))**2).sum()*252/(len(S_t)-1))

"""
Variance swap payout, specify either vega notional or variance notional
"""
def variance_swap_payout(S_t, K, var_n=None, vega_n=None, cap=None):
    if vega_n is not None:
        var_n = vega_n / (2*K) * 100
    if cap is not None:
        return (min(realized_vol(S_t)**2,(cap*K)**2) - K**2)*var_n
    return (realized_vol(S_t)**2 - K**2)*var_n

"""
Basic Black Scholes valuation/greeks
K   : strike
T   : years to expiry
pc  : 'P' if put, 'C' if call
S0  : spot
r   : risk-free rate
q   : cost-of-carry (i.e. dividend + borrow)
vol : annualized implied volatility
"""
def BS_value(K, T, pc, S0, r, q, vol):
    if T == 0:
        return max(S0 - K, 0) if pc == 'C' else max(K - S0, 0)
    d1 = (np.log(S0/K) + (r - q + vol**2/2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    if pc == 'P':
        return np.exp(-r*T)*stat.norm.cdf(-d2)*K - stat.norm.cdf(-d1)*S0
    else:
        return stat.norm.cdf(d1)*S0 - np.exp(-r*T)*stat.norm.cdf(d2)*K

def BS_delta(K, T, pc, S0, r, q, vol):
    d1 = (np.log(S0/K) + (r - q + vol**2/2)*T)/(vol*np.sqrt(T))
    return stat.norm.cdf(d1) if pc == 'C' else -stat.norm.cdf(-d1)
