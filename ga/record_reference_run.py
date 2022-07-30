"""
utiilty for creating the reference value of various monitoring values. They can then be used in the genetic optimization's fitness function.
"""

from os.path import join, isfile
from time import time as t
import subprocess
import ga_utility
import json

# path to the reference run config file
REF_CONFIG = join("configs", "run", "test.json")
# make sure this match config file result path
RESULT_PATH = "output_tmp"

# observables, each should appear in all metrics
observables = [
    "th_pt",
    "th_y",
    "th_phi",
    "th_e",
    "tl_pt",
    "tl_y",
    "tl_phi",
    "tl_e"
]

ref_dict = {}

# edit here if unfolder path is changed
start = t()
subprocess.run(["./run_unfold.py", REF_CONFIG])
time = t() - start

# extract and record monitoring values
ref_dict["time"] = time
for observable in observables:
    ref_dict[observable+"_pvalue"] = ga_utility.extract_nominal_pval(observable, RESULT_PATH)[-1]
    ref_dict[observable+"_delta_std"] = ga_utility.extract_rerun_delta_std(observable, RESULT_PATH)[-1]

ref = join("ga", "ref.json")
if not isfile(ref):
    with open(ref, "x") as file:
        pass

with open(ref, "w") as file:
    json.dump(ref_dict, file, indent="")