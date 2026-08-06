import datetime as dt
import matplotlib.pyplot as plt
import random as rand
import sys
import indicators as ind
import marketsimcode as sim
import ManualStrategy as ms
import experiment1 as e1
import experiment2 as e2

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


def manual_strategy ():
    # Variables
    impact = 0.005
    commission = 9.95
    sv=100000
    symbol="JPM"
    sd_train=dt.datetime(2008, 1, 1)
    ed_train=dt.datetime(2009, 1, 1)
    sd_test=dt.datetime(2010, 1, 1)
    ed_test=dt.datetime(2011, 1, 1)
    verbose = False

    ###### Manual Strategy ######
    ms_trades_train= ms.testPolicy(symbol, sd_train, ed_train, sv) # in sample
    ms_trades_test = ms.testPolicy(symbol, sd_test, ed_test, sv) # out of sample

    ###### Benchmark ######
    df_bench_train = ind.benchmark(symbol, sd_train, ed_train) # in sample benchmark
    df_bench_test = ind.benchmark(symbol, sd_test, ed_test) # out of sample benchmark

    ###### In sample portfolio value ######
    pv_ms_train = sim.compute_portvals(ms_trades_train, symbol, sv, commission, impact)
    pv_bench_train = sim.compute_portvals(df_bench_train, symbol, sv, commission, impact)

    ###### Out of sample portfolio value ######
    pv_ms_test = sim.compute_portvals(ms_trades_test, symbol, sv, commission, impact)
    pv_bench_test = sim.compute_portvals(df_bench_test, symbol, sv, commission, impact)

    ###### Manual Strategy Plot ######
    # MANUAL in sample
    plt.clf()
    plt.plot(pv_ms_train / pv_ms_train.iloc[0], label="Manual Strategy", color="red")
    plt.plot(pv_bench_train / pv_bench_train.iloc[0], label="Benchmark", color="purple")
    first_short = True
    first_long = True
    for date in ms_trades_train.index:
        if ms_trades_train.loc[date].iloc[0] <= -1:
            plt.axvline(x=date, label="Short"if first_short else None, color="black")
            first_short = False
        if ms_trades_train.loc[date].iloc[0] >= 1:
            plt.axvline(x=date, label="Long" if first_long else None, color="blue")
            first_long = False
    plt.title("Manual Strategy: In-sample testing")
    plt.legend()
    plt.savefig("./images/manual strategy in-sample.png")
    # plt.show()

    # MANUAL out of sample
    plt.clf()
    plt.plot(pv_ms_test / pv_ms_test.iloc[0], label="Manual Strategy", color="red")
    plt.plot(pv_bench_test / pv_bench_test.iloc[0], label="Benchmark", color="purple")
    first_short = True
    first_long = True
    for date in ms_trades_test.index:
        if ms_trades_test.loc[date].iloc[0] <= -1:
            plt.axvline(x=date, label="Short"if first_short else None, color="black")
            first_short = False
        if ms_trades_test.loc[date].iloc[0] >= 1:
            plt.axvline(x=date, label="Long" if first_long else None, color="blue")
            first_long = False
    plt.title("Manual Strategy: Out-of-sample testing")
    plt.legend()
    plt.savefig("./images/manual strategy out-of-sample.png")
    # plt.show()



if __name__ == "__main__":
    rand.seed(904199288)
    sys.stdout=open("output.txt","w")
    manual_strategy()
    e1.experiment1()
    e2.experiment2()
    sys.stdout.close()