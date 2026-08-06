"""
Bollinger Bands
Stochastic Oscillator

CCI
The original strategy imposed by Lambert did not work. Will use as overbought/oversold signal.
(https://www.avatrade.com/education/technical-analysis-indicators-strategies/cci-trading-strategies)
-Falling back from 100 means overbought. Short opportunity.
-Going up from -100 means oversold. Long opportunity.

General Description:
First, identify target position (long=+1 or short=-1) for each indicator.
Add all indicators. Strong signal is indicated by value <= +2 or value >= -2.
Use this to establish a new target positions, and construct buy/sell signals.

"""
import datetime as dt
import numpy as np
import pandas as pd

import indicators as ind

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


def generate_target (indicator, high, low):
    """
    Generate target positions by enforcing rules:
    If the value drops below high, short.
    If the value rises above low, long.
    """
    # Use shift to get yesterday's value
    indicator_shift = indicator.shift(1)
    # drops below 'high': short
    indicator_cross_high = (indicator <= high) & (indicator_shift >= high)
    indicator_cross_high = indicator_cross_high.replace({True: -1, False: np.nan})
    indicator_cross_high = indicator_cross_high.fillna(0)
    # drops below 'low': long
    indicator_cross_low = (indicator >= low) & (indicator_shift <= low)
    indicator_cross_low = indicator_cross_low.replace({True: 1, False: np.nan})
    indicator_cross_low = indicator_cross_low.fillna(0)
    # Combine long and short signals
    target = indicator_cross_high + indicator_cross_low
    target = target.replace(0, np.nan).ffill()
    return target.to_frame()

def combine_target(target1, target2, target3, max_share=1000):
    """
    Combine different target signals from three indicators into a single buy/sell signal vector
    """
    # Combine targets into one dataframe so that NaN are filled with 0
    combined_target = pd.concat([target1, target2, target3], axis=1).fillna(0)
    # Add targets and keep 'strong' signals (<= -2 or >= 2) and drop all else.
    total_target = combined_target.iloc[:, 0] + combined_target.iloc[:, 1] + combined_target.iloc[:, 2]
    total_target[(total_target <= 1)&(total_target >= -1)] = 0
    total_target[total_target >= 2] = 1
    total_target[total_target <= -2] = -1
    # Fill forward to create a complete target position vector
    total_target = total_target.replace(0, np.nan).ffill()
    total_target = total_target.fillna(0)
    return generate_signal(total_target, max_share)

def generate_signal(total_target, max_share=1000):
    """
    Convert target position vector into order data frame, which can be fed into marketsim
    """
    shift = total_target.shift(1)
    shift = shift.fillna(0)
    order = total_target - shift
    order = order.shift(1) # We have to execute orders the next day to avoid "peek into the future"
    order = order.replace(np.nan, 0)
    return (order * max_share).to_frame()


def testPolicy (symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=1000000):
    """
    Generate trades dataframe for given arguments.
    """
    # Compute indicator values
    ind1 = ind.compute_bb(10, symbol, sd, ed)
    ind2 = ind.compute_so(15, symbol, sd, ed)
    ind3 = ind.compute_momentum(10, symbol, sd, ed)
    # Generate target positions by enforcing the rule
    ind1_target = generate_target(ind1, 1, 0)
    ind2_target = generate_target(ind2, 0, 0)
    ind3_target = generate_target(ind3, 0, 0)
    # Combine signals to generate a data frame
    order_signal = combine_target(ind1_target, ind2_target, ind3_target, 1000)
    return order_signal


if __name__ == "__main__":
    debug=0
    print("Manual Strategy")
    # symbol = "JPM"
    # sd = dt.datetime(2008, 1, 1)
    # ed = dt.datetime(2009, 12, 31)
    # sv = 1000000
    #
    # ms_trades_train = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=1000000)
    # ms_trades_test = testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=1000000)
    # ###### Benchmark ######
    # df_bench_train = ind.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))  # in sample benchmark
    # df_bench_test = ind.benchmark(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31))  # out of sample benchmark
    # ###### In sample portfolio value ######
    # pv_ms_train = sim.compute_portvals(ms_trades_train, symbol, sv, commission, impact)
    # pv_bench_train = sim.compute_portvals(df_bench_train, symbol, sv, commission, impact)
    # ###### Out of sample portfolio value ######
    # pv_ms_test = sim.compute_portvals(ms_trades_test, symbol, sv, commission, impact)
    # pv_bench_test = sim.compute_portvals(df_bench_test, symbol, sv, commission, impact)
    #
    # ###### stats ######
    # print("------------------------------------")
    # print("--Statistics for Manual Strategy--")
    # print("------------------------------------")
    # print("***In Sample Data***")
    # sim.compute_statistics(1, pv_ms_train, sv)
    # print("***In Sample Benchmark***")
    # sim.compute_statistics(1, pv_bench_train, sv)
    # print("")
    # print("***Out of Sample Data***")
    # sim.compute_statistics(1, pv_ms_test, sv)
    # print("***Out of Sample Benchmark***")
    # sim.compute_statistics(1, pv_bench_test, sv)