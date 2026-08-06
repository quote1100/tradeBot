import datetime as dt
import matplotlib.pyplot as plt
import indicators as ind
import marketsimcode as sim
import ManualStrategy as ms
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


def experiment1():
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

    ###### Q LEARNER ######
    # Train
    learner = sl.StrategyLearner(verbose, impact, commission) # constructor
    learner.add_evidence(symbol, sd_train, ed_train, sv) # training phase
    # Test
    sl_trades_train = learner.testPolicy(symbol, sd_train, ed_train, sv) # in sample
    sl_trades_test = learner.testPolicy(symbol, sd_test, ed_test, sv) # out of sample

    ###### Benchmark ######
    df_bench_train = ind.benchmark(symbol, sd_train, ed_train) # in sample benchmark
    df_bench_test = ind.benchmark(symbol, sd_test, ed_test) # out of sample benchmark

    ###### In sample portfolio value ######
    pv_str_train = sim.compute_portvals(sl_trades_train, symbol, sv, commission, impact)
    pv_ms_train = sim.compute_portvals(ms_trades_train, symbol, sv, commission, impact)
    pv_bench_train = sim.compute_portvals(df_bench_train, symbol, sv, commission, impact)

    ###### Out of sample portfolio value ######
    pv_str_test = sim.compute_portvals(sl_trades_test, symbol, sv, commission, impact)
    pv_ms_test = sim.compute_portvals(ms_trades_test, symbol, sv, commission, impact)
    pv_bench_test = sim.compute_portvals(df_bench_test, symbol, sv, commission, impact)

    ###### stats ######

    print("------------------------------------")
    print("------------EXPERIMENT 1------------")
    print("------------------------------------")
    print("")
    print("------------------------------------")
    print("Statistics for Benchmark")
    print("------------------------------------")
    print("***In Sample Data***")
    sim.compute_statistics(1, pv_bench_train, sv)
    print("")
    print("***Out of Sample Data***")
    sim.compute_statistics(1, pv_bench_test, sv)
    print("")
    print("------------------------------------")
    print("Statistics for Manual Strategy")
    print("------------------------------------")
    print("***In Sample Data***")
    sim.compute_statistics(1, pv_ms_train, sv)
    print("")
    print("***Out of Sample Data***")
    sim.compute_statistics(1, pv_ms_test, sv)
    print("")
    print("------------------------------------")
    print("Statistics for Strategy Learner")
    print("------------------------------------")
    print("***In Sample Data***")
    sim.compute_statistics(1, pv_str_train, sv)
    print("")
    print("***Out of Sample Data***")
    sim.compute_statistics(1, pv_str_test, sv)

    ###### Plot ######
    # Plot: in sample
    plt.clf()
    plt.plot(pv_str_train / pv_str_train.iloc[0], label="Strategy Learner", color="green")
    plt.plot(pv_ms_train / pv_ms_train.iloc[0], label="Manual Strategy", color="red")
    plt.plot(pv_bench_train / pv_bench_train.iloc[0], label="Benchmark", color="purple")
    plt.title("Experiment 1: In-sample Testing")
    plt.legend()
    plt.savefig("./images/experiment1 in-sample.png")
    # plt.show()

    # Plot: out of sample
    plt.clf()
    plt.plot(pv_str_test / pv_str_test.iloc[0], label="Strategy Learner", color="green")
    plt.plot(pv_ms_test / pv_ms_test.iloc[0], label="Manual Strategy", color="red")
    plt.plot(pv_bench_test / pv_bench_test.iloc[0], label="Benchmark", color="purple")
    plt.title("Experiment 1: Out-of-sample Testing")
    plt.legend()
    plt.savefig("./images/experiment1 out-of-sample.png")
    # plt.show()

if __name__ == "__main__":
    experiment1()
    print("Experiment 1")
