""""""
"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Erich Sung
GT User ID: esung39
GT ID: 904199288
"""

import datetime as dt
import pandas as pd
import numpy as np

import util as ut
import QLearner as ql
import indicators as ind


"""
Purpose: 
    This project uses regression Learner (Q-Learner) to learn policies based on three indicators: Bollinger Bands, 
    Stochastic Oscillator, and CCI, which determines when to long/short a stock to optimize the return.
State: 
    Indicators and current position. 
    *current position is used to identify position change, which applies impact/commission to reward calculation. 
Actions: 
    Short, long, or out
Reward: 
    Return due to price difference between today and tomorrow's price.
"""


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.mean = None
        self.std = None


    def normalize(self, indicator):
        """
        Normalizes indicators and save normalizing factors (mean and std) so that it can be reused for testing.
        """
        if self.mean is None and self.std is None:
            self.mean = indicator.mean()
            self.std = indicator.std()
        return (indicator - self.mean) / self.std

    def discretize(self, indicators, bin_num=6):
        """
        Purpose:
            Convert indicators and position (=3) into a single integer
            position of the current state can be: [0=short, 1=out, 2=long]
        Equation:
            discretized = (indicator1 * 3 * bin^2) + (indicator2 * 3 * bin) + (indicator3 * 3) + 3
        Parameters:
            indicators: dataFrame containing indicator value. Uses first three columns.
            bin_num: number of bins (or levels) for the indicator. Reference below.
        Reference:
            https://pandas.pydata.org/docs/reference/api/pandas.cut.html
        """
        norm_state = self.normalize(indicators.iloc[:,0:-1])
        bin_range = [-np.inf, -1, -0.25, 0, 0.25, 1, np.inf]
        norm_state = norm_state.apply(pd.cut, bins=bin_range, labels=False)

        return (norm_state.iloc[:,0] * 3 * bin_num * bin_num) + (norm_state.iloc[:,1] * 3 * bin_num) + (norm_state.iloc[:,2] * 3)


    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
        epochs = 200,
        max_position = 1000
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # Reset mean/std in case learning another sample.
        self.mean = None
        self.std = None

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Create learner object.
        self.learner = ql.QLearner(
            num_states=648, # each indicator has 6 bins: 3*6^3
            num_actions=3, # long, short, out
            alpha=0.1,
            gamma=0.8,
            rar=0.8,
            radr=0.99998, # Important to adjust according to the sample size.
            dyna=0,
            verbose=False,
        )

        # Compute indicators. (Bollinger bands, stochastic oscillator, and CCI)
        ind1 = ind.compute_bb(10, symbol, sd, ed)
        ind2 = ind.compute_so(15, symbol, sd, ed)
        ind3 = ind.compute_momentum(10, symbol, sd, ed)
        indicators = pd.concat([ind1.rename("ind1"), ind2.rename("ind2"), ind3.rename("ind3"), prices.iloc[:]], axis=1)
        indicators = indicators.dropna()
        discretized_state = self.discretize(indicators) # Discretize indicators into a single int value.

        # start learning
        scores = np.zeros((epochs, 1))
        for epoch in range(1, epochs + 1):
            # Initialize variables
            total_reward = 0
            action = 1
            count = 0
            data = indicators.iloc[:,-1].copy()

            # Set initial state and get action
            state = discretized_state.iloc[0] + action
            action = self.learner.querysetstate(state)

            """
            Note:
            When we apply the learner, we're basically executing trades using yesterday's indicators because  
            trades must be done on days that follow the latest close.
            The problem we are trying to solve then is: 
            
            "based on today's indicators, what is the optimal reward by taking an action tomorrow?"
            In order to solve this, we need to estimate the price the next day and the day after.
            the earning would be the price difference between the two days.
            """
            # Iterate through the price data
            for i in range(data.shape[0]-5):
                # Get current position to see if position change is expected.
                current_position = state % 3
                position_change = abs(current_position - action)
                # Calculate reward. Impact and commission apply if there's position change
                earning = (action - 1) * (data.iloc[i + 2] - data.iloc[i + 1]) # IMPORTANT: trade happens a day after closing.
                if position_change != 0:
                    r = max_position * earning - max_position * position_change * data.iloc[i+1] * self.impact - self.commission
                else:
                    r = max_position * earning
                # set next state using current action, then get next action from the next state.
                next_state = discretized_state.iloc[i + 1] + action
                next_action = self.learner.query(next_state, r)
                state = next_state
                action = next_action
                # statistics
                total_reward += r
                count = count + 1
            if count == 100000:
                print("timeout")
            if self.verbose:
                print("Learning epochs:", epoch, "/", epochs)
                print(f"{epoch}, {total_reward}")
            scores[epoch - 1, 0] = total_reward
        if self.verbose:
            print("Learning complete")
        return np.median(scores)


    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 1, 1),
        sv=10000,
        max_position=1000
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)

        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        prices = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Compute indicators exactly as it was in training. (Important: use same normalizing factors and bins)
        ind1 = ind.compute_bb(10, symbol, sd, ed)
        ind2 = ind.compute_so(15, symbol, sd, ed)
        ind3 = ind.compute_momentum(10, symbol, sd, ed)
        indicators = pd.concat([ind1.rename("ind1"), ind2.rename("ind2"), ind3.rename("ind3"), prices.iloc[:]], axis=1)
        indicators = indicators.dropna()
        discretized_state = self.discretize(indicators) # Important: use same normalizing factors and bins

        # Set initial variables
        action = 1
        target = indicators.iloc[:,3] * 0
        trades = prices.iloc[:, 0]*0

        # Get state and action based on the indicators
        for i in range(discretized_state.shape[0] - 1):
            state = discretized_state.iloc[i] + action
            q_row = self.learner.Q[state]
            if np.all(q_row == q_row[0]):  # all values equal 0: never meaningfully updated.
                action = action  # exit position for unknown (unexpected) cases.
            else:
                action = np.argmax(q_row)
            target.values[i] = (action - 1) # set target position (action - 1: -1=short, 0=out, +1=long)

        # Based on target positions, compute trades dataframe.
        shift = target.shift(1)
        shift = shift.fillna(0)
        target = target - shift
        target = target.replace(np.nan, 0)
        target = target.shift(1)  # Since we don't want to "peek into the future", we have to execute orders the next day.
        trades = (trades + (target * max_position)).replace(np.nan, 0).to_frame()

        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)
        if self.verbose:
            print(prices_all)
        return trades

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "esung39"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        """
        A comma separated string of GT_Name of each member of your study group
        # Example: "gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
        """
        return "esung39"


if __name__ == "__main__":
    # Uncomment for testing with default values
    # rand.seed(904199288)
    # a = StrategyLearner()
    # a.add_evidence()
    # a.testPolicy()
    print("One does not simply think up a strategy")

