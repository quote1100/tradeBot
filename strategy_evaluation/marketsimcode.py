import pandas as pd
from util import get_data, plot_data


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


def compute_portvals(
    orders,
    symbol,
    start_val=1000000,
    commission=0.0,
    impact=0.0,
):
    """
    Computes the portfolio values with input dataframe
    """

    # Set all variables from orders.
    start_date = orders.index.min()
    end_date = orders.index.max()

    # Set up prices dataframe using criteria from order file.
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[[symbol]]
    cash_col = 'CASH'
    prices[cash_col] = 1.0 # later used to evaluate portfolio value.

    # Initialize trades dataframe. This will hold transaction information.
    trades = prices.copy() * 0

    # Evaluate transaction cost depending on BUY or SELL.
    for i in range (orders.shape[0]):
        date = orders.index[i]
        shares = orders[symbol].iloc[i]

        if shares == 0:
            trades.loc[date, symbol] += 0
            trades.loc[date, cash_col] += 0
        elif shares != 0:
            transaction_cost = prices.loc[date, symbol] * (1 + impact) * shares - commission
            trades.loc[date, symbol] += shares
            trades.loc[date, cash_col] += -transaction_cost
        else:
            print("ERROR: UNKNOWN VALUE FOR ORDER. MUST BE EITHER BUY OR SELL")
            exit()

    # Create holdings dataframe. This holds balance of stocks and cash in possession.
    holdings = trades.copy()
    holdings.iloc[0][cash_col] += start_val
    holdings = holdings.cumsum()

    # Evaluate portfolio values. Essentially multiplies stock price and shares
    values = prices * holdings

    # Evaluate total portfolio value into a single column dataframe
    rv = pd.DataFrame(values.sum(axis=1), columns=["VALUE"])

    return rv


def compute_statistics(allocs, prices, start_value):
    # Compute portfolio value
    normalized_data = prices / prices.iloc[0]
    alloced_data = normalized_data * allocs
    position_value = alloced_data * start_value
    portfolio_value = position_value.sum(axis = 1)
    index_today = portfolio_value.shape[0]

    # print(alloced_data)

    # Compute daily return
    daily_return = (portfolio_value / portfolio_value.shift(1)) - 1
    daily_return = daily_return.dropna() # Drop the first day

    # Compute average daily return, standard deviation, and cumulative return
    average_daily_return = daily_return.mean()
    standard_deviation = daily_return.std(ddof=1)
    cumulative_return = (portfolio_value.iloc[index_today - 1] / portfolio_value.iloc[0]) - 1

    # Compute Sharpe ratio.
    if standard_deviation == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = (252**0.5) * average_daily_return / standard_deviation

    print(f"Cumulative return: {cumulative_return:6f}")
    print(f"Average Daily return: {average_daily_return:6f}")
    print(f"Standard Deviation: {standard_deviation:6f}")
    print(f"Sharpe Ratio: {sharpe_ratio:6f}")

    return cumulative_return, average_daily_return, standard_deviation, sharpe_ratio


if __name__ == "__main__":
    compute_portvals()
    print("Hello World")
