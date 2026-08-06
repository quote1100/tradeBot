import pandas as pd
import numpy as np

from util import get_data, plot_data


"""
Indicators  
1. Bollinger Bands
2. Simple Moving Average (SMA)
3. momentum
4. Stochastic Oscillator
5. Commodity Channel Index (CCI)
"""


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "esung39"  # replace tb34 with your Georgia Tech username.

def study_group():
    """
    A comma separated string of GT_Name of each member of your study group
    # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
    """
    return "esung39"


def compute_bb(window=15, symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    """
    Bollinger Bands
    Using Balch's method: Buy if the prices goes below the lower band and comes back into the band.
    Sell if the price goes above the upper band and comes back.
    These points can be identified using BBP or %B.
    If BBP = 1, it touches upper band, if BBP = 0 it touches lower band
    """
    # Retrieve data frame
    date_range_band = pd.date_range(sd, ed)
    df_bb = get_data([symbol], date_range_band)
    df_bb = df_bb[symbol]

    # Normalize and Compute SMA, STD, and BB
    sma_bb = df_bb.rolling(window=window).mean()
    std = df_bb.rolling(window=window).std()
    upper_bb = sma_bb + std * 2
    lower_bb = sma_bb - std * 2

    # Compute BBP (%B)
    bbp = (df_bb - lower_bb)/(upper_bb - lower_bb)
    bbp = bbp.dropna()

    return bbp


def compute_sma(window=20, symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    """
    Similar to Bollinger Bands.
    If the price moves above a moving average: buy
    If the price moves below the moving average: short
    """
    # Retrieve data frame
    date_range_band = pd.date_range(sd, ed)
    df_sma = get_data([symbol], date_range_band)
    df_sma = df_sma[symbol]

    # Normalize and compute SMA
    df_sma = df_sma/df_sma[0]
    sma_sma = df_sma.rolling(window=window).mean()
    price_to_sma = df_sma/sma_sma

    return price_to_sma


def compute_momentum(n=10, symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    """
    Momentum shows change of price
    Momentum swing is the indicator. If moving below 0, then short.
    If moving above 0, then long.
    """
    # Retrieve data frame
    date_range_mo = pd.date_range(sd, ed)
    df_mo = get_data([symbol], date_range_mo)
    df_mo = df_mo[symbol]

    # Normalize and compute momentum
    df_mo = df_mo/df_mo[0]
    momentum = (df_mo/df_mo.shift(n)) - 1
    momentum = momentum.dropna()

    return momentum


def compute_so(window=14, symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    """
    First compute faster %K and slower %D
    2 signals
    1. %K drops below 80, sell.  %K moves above 20, buy
    2. %K crosses %D
    take 2. approach, which is stochastic crossover
    """
    # Retrieve data frame
    date_range_st = pd.date_range(sd, ed)
    df_st = get_data([symbol], date_range_st)
    df_st = df_st[symbol]

    # Compute rolling min/max
    rolling_min = df_st.rolling(window).min()
    rolling_max = df_st.rolling(window).max()

    # Compute %K and %D
    k = ((df_st-rolling_min)/(rolling_max-rolling_min))
    d = k.rolling(3).mean()
    # Changed from k/d to k-d. Need to avoid diving by 0.
    k_to_d = k - d

    return k_to_d


def compute_cci(window=20, symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    """
    CCI
    Used typical price instead of closing
    If goes above 100, long. If comes back to 100, exit.
    If goes below -100, short. If comes back to -100, exit.
    """
    # Retrieve data frame
    date_range_st = pd.date_range(sd, ed)
    df_high = get_data([symbol], date_range_st, True, "High")
    df_high = df_high[symbol]
    df_low = get_data([symbol], date_range_st, True, "Low")
    df_low = df_low[symbol]
    df_close = get_data([symbol], date_range_st, True, "Close")
    df_close = df_close[symbol]

    # Compute typical price
    df_cci = (df_high + df_low + df_close)/3

    # Compute SMA, MAD to compute CCI
    sma_cci = df_cci.rolling(window).mean()
    mad = df_cci.rolling(window).apply(mean_deviation, True)
    cci = (df_cci-sma_cci)/(0.015*mad)

    return cci

def mean_deviation(x):
    """
    Adapted from Groette. O, "A Python CCI Trading Strategy (Backtest)"
    https://www.quantifiedstrategies.com/cci-trading-strategy-python/
    """
    return np.mean(np.abs(x-np.mean(x)))

def benchmark(symbol='JPM', sd='2008-01-01', ed='2009-12-31'):
    # Retrieve data frame
    date_range_st = pd.date_range(sd, ed)
    df = get_data([symbol], date_range_st)
    df = df[symbol]

    return_df = pd.DataFrame({symbol: [1000]}, index=df.index[:])
    return_df.iloc[1:] = 0
    return return_df


if __name__ == "__main__":
    # Testing Stock
    start_date = '2008-01-01'
    end_date = '2009-12-31'
    symbol = 'JPM'
    date_range = pd.date_range(start_date, end_date)
    test_df = get_data([symbol], date_range)
    test_df = test_df[symbol]


    compute_sma(20, symbol, start_date, end_date)
    compute_bb(15, symbol, start_date, end_date)
    compute_momentum(10, symbol, start_date, end_date)
    compute_so(14, symbol, start_date, end_date)
    compute_cci(20, symbol, start_date, end_date)



