"""
learning rate: 0.001, 0.005, 0.025, 0.125
iterations: 1, 3, 9
model depth: 1, 5, 10
model width: 25, 50, 100
extra model: 200x5, 1000x1, 10000x1
N = 20
"""

import json
from time import time as t
import tracemalloc as m
import gc
import subprocess

TEST_PLAN = "test_plan.json"
RESULT_PATH = "/home/xyyu/omnifoldTop/test_results/omniTest.json"
CONFIG_PATH = "configs/run/_test_auto.json"
TEST_DIR = "/fast_scratch/xyyu/model_learning_iteration_test_data/"
# TEST_DATA = [
#     "ttbar_1_pseudotop_parton_ejets.root",
#     "ttbar_5_pseudotop_parton_ejets.root"
#     ]
# TEST_SIG = [
#     "ttbar_hw_3_pseudotop_parton_ejets.root",
#     "ttbar_hw_6_pseudotop_parton_ejets.root"
# ]

TEST_DATA = [
    "ttbar_5_pseudotop_parton_ejets.root"
    ]
TEST_SIG = [
    "ttbar_hw_6_pseudotop_parton_ejets.root"
]

def write_complete_flag():
    """
    write test complete status to test plan json file
    """
    file = open(TEST_PLAN, "w")
    json.dump(test_dict, file, indent="")
    file.close()

def write_complete(name):
    """
    mark a test as complete and save to json
    """
    test_dict[name] = 1
    write_complete_flag()

def commence_test(name):
    """
    write setting to config and call run_unfold

    Returns
    -------
    time took to run the test
    """
    t_i = t()
    subprocess.run(["./run_unfold.py","configs/run/_test_auto.json"])
    t_f = t()
    return t_f - t_i

def generate_config(name):
    """
    generate tets config as dictionary
    """
    config = dict()
    config["data"] = [TEST_DIR + path for path in TEST_DATA]
    config["signal"] = [TEST_DIR + path for path in TEST_SIG]
    config["observable_config"] = "configs/observables/vars_ttbardiffXs_pseudotop.json"
    config["truth_known"] = True
    config["normalize"] = True
    config["batch_size"] = 32768 # 2**15
    config["plot_verbosity"] = 3
    config["outputdir"] = "output_tmp"
    config["nresamples"] = 20
    config["error_type"] = "bootstrap_full"
    
    # format specific parsing, see init_test.py
    config["iterations"] = int(name[name.find("i") + 1:name.find("m")])
    config["model_name"] = name[name.find("m") + 1:]
    config["learning_rate"] = float(name[name.find("l") + 1:name.find("i")])
    return config

def write_config(name):
    """
    write test config to json
    """
    file = open(CONFIG_PATH, "w")
    json.dump(generate_config(name), file, indent="")
    file.close()

def log_result(name, time, mcurrent, mpeak):
    """
    record test results
    """
    model_result = dict()
    model_result["observables"] = dict()
    vars = ['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e']
    files = ["output_tmp/Metrics/"+ var + ".json" for var in vars]
    for var in vars:
        metric_path = "output_tmp/Metrics/"+ var + ".json"
        metric_file = open(metric_path, "r")
        metric_dict = json.load(metric_file)
        iterations = metric_dict[var]["nominal"]["Chi2"]["iterations"]
        chi2ndf = metric_dict[var]["nominal"]["Chi2"]["chi2/ndf"]
        model_result["observables"][var] = dict()
        model_result["observables"][var]["iterations"] = iterations
        model_result["observables"][var]["chi2/ndf"] = chi2ndf
    model_result["time"] = time
    model_result["mexit"] = mcurrent
    model_result["mpeak"] = mpeak
        
    result_dict[name] = model_result

    file = open(RESULT_PATH, "w")
    json.dump(result_dict, file, indent="   ")
    file.close()

file = open(TEST_PLAN, "r")
try:
    test_dict = json.load(file)
except json.decoder.JSONDecodeError:
    print("Error in reading test plan. Make sure to run init_test.py first")
    test_dict = dict()
file.close()

file = open(RESULT_PATH, "r")
try:
    result_dict = json.load(file)
except json.decoder.JSONDecodeError:
    print("Possibly empty result file, creating new")
    result_dict = dict()
file.close()

# commence the test (skip completed ones)

for test_name in test_dict:
    if test_dict[test_name] == 0:
        print("Commencing test:{0}".format(test_name))
        write_config(test_name) # temp
        m.start()
        time = commence_test(test_name)
        mcurrent, mpeak = m.get_traced_memory()
        log_result(test_name, time, mcurrent, mpeak)
        write_complete(test_name)
        gc.collect()
        print("Test {0} complete. Written result to file.".format(test_name))
print("All test completed")
