import datetime as dt
import matplotlib.pyplot as plt
import indicators as ind
import marketsimcode as sim
import StrategyLearner as sl

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


def experiment2_compute(impact = 0, commission = 0, title=""):
    # Variables

    sv=100000
    symbol="JPM"
    sd_train=dt.datetime(2008, 1, 1)
    ed_train=dt.datetime(2009, 1, 1)
    sd_test=dt.datetime(2010, 1, 1)
    ed_test=dt.datetime(2011, 1, 1)
    verbose = False

    ###### Q LEARNER ######
    # Train
    learner = sl.StrategyLearner(verbose, impact, commission) # constructor
    learner.add_evidence(symbol, sd_train, ed_train, sv) # training phase
    # Test
    sl_trades_train = learner.testPolicy(symbol, sd_train, ed_train, sv) # in sample
    sl_trades_test = learner.testPolicy(symbol, sd_test, ed_test, sv) # out of sample

    ###### Benchmark ######
    df_bench_train = ind.benchmark(symbol, sd_train, ed_train) # in sample benchmark
    # df_bench_test = ind.benchmark(symbol, sd_test, ed_test) # out of sample benchmark

    ###### In sample portfolio value ######
    pv_str_train = sim.compute_portvals(sl_trades_train, symbol, sv, commission, impact)
    # pv_bench_train = sim.compute_portvals(df_bench_train, symbol, sv, commission, impact)

    ###### Out of sample portfolio value ######
    pv_str_test = sim.compute_portvals(sl_trades_test, symbol, sv, commission, impact)
    # pv_bench_test = sim.compute_portvals(df_bench_test, symbol, sv, commission, impact)

    ###### stats ######
    print("------------------------------------")
    print("Statistics for Strategy Learner")
    print(title)
    print("------------------------------------")
    print("***In Sample Data***")
    sim.compute_statistics(1, pv_str_train, sv)
    print("")
    print("***Out of Sample Data***")
    sim.compute_statistics(1, pv_str_test, sv)
    print("")

    return pv_str_train, pv_str_test

def experiment2():
    print("")
    print("------------------------------------")
    print("------------EXPERIMENT 2------------")
    print("------------------------------------")
    print("")
    pv_zero_train, pv_zero_test = experiment2_compute(0, 0, "Impact=0")
    pv_low_train, pv_low_test = experiment2_compute(0.005, 0, "Impact = 0.005")
    pv_high_train, pv_high_test = experiment2_compute(0.05, 0, "Impact=0.05")

    ###### Plot ######
    # Plot: in sample
    plt.clf()
    plt.plot(pv_zero_train / pv_zero_train.iloc[0], label="Impact=0")
    plt.plot(pv_low_train / pv_low_train.iloc[0], label="Impact=0.005")
    plt.plot(pv_high_train / pv_high_train.iloc[0], label="Impact=0.05")
    plt.title("Experiment 2: In-sample Testing")
    plt.legend()
    plt.savefig("./images/experiment2 in-sample.png")
    # plt.show()

    # Plot: out of sample
    plt.clf()
    plt.plot(pv_zero_test / pv_zero_test.iloc[0], label="Impact=0")
    plt.plot(pv_low_test / pv_low_test.iloc[0], label="Impact=0.005")
    plt.plot(pv_high_test / pv_high_test.iloc[0], label="Impact=0.05")
    plt.title("Experiment 2: Out-of-sample Testing")
    plt.legend()
    plt.savefig("./images/experiment2 out-of-sample.png")
    # plt.show()

if __name__ == "__main__":
    experiment2()
