"""
utiilty for creating the reference value of various monitoring values. They can then be used in the genetic optimization's fitness function.
"""

from os.path import join
import time.time as t
import subprocess

# path to the reference run config file
REF_CONFIG = join("configs", "run", "quick_run.json")

ref_dict = {}

# edit here if unfolder path is changed
start = t()
subprocess.run(["./run_unfolder.py", REF_CONFIG])
time = t() - start

# extract and record monitoring values
ref_dict["time"] = time