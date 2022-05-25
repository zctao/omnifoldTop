#!/usr/bin/env python3
import os
import sys
import glob
import util
from unfoldv2 import unfold

# Usage
if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("Usage: ./replot_unfold.py <path_to_result> [<new_output_dir>]")
    sys.exit(1)

fpath_result_dir = sys.argv[1]
new_outdir = sys.argv[2] if len(sys.argv) == 3 else None 

# check if directory exists
if not os.path.isdir(fpath_result_dir):
    print(f"ERROR: cannot find directory {fpath_result_dir}")
    sys.exit(2)

# run argument json
fpath_args = os.path.join(fpath_result_dir, "arguments.json")
if not os.path.isfile(fpath_args):
    print(f"ERROR: cannot find argument configuration for the previous run: {fpath_args}")
    sys.exit(2)

run_cfg = util.read_dict_from_json(fpath_args)
if not run_cfg:
    print(f"ERROR: fail to load run config from {fpath_args}")
    sys.exit(3)

# use unfolded weights
fpaths_uw = glob.glob(os.path.join(fpath_result_dir, "weights*.npz"))
fpaths_uw.sort()

run_cfg.update({"unfolded_weights": fpaths_uw})

# if new output directory is provided:
if new_outdir is not None:
    if not os.path.isdir(new_outdir):
        print("Create new output directory")
        os.makedirs(new_outdir)

    run_cfg["outputdir"] = new_outdir

# Rerun
print("Rerun plotting")
unfold(**run_cfg)
