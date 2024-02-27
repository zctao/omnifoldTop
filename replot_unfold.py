#!/usr/bin/env python3
import os
import sys
import glob
import util
from unfoldv2 import unfold

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('result_dir', type=str, help="Unfolding result directory")
parser.add_argument('-o', '--outdir-new', type=str,
                    help="New output directory. If not provided, use the unfolding result directory")
parser.add_argument('--no-ibu', action='store_true', help="If True, do not plot IBU")

args = parser.parse_args()

fpath_result_dir = args.result_dir

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
fpaths_uw = glob.glob(os.path.join(fpath_result_dir, "weights*.h5"))
if not fpaths_uw:
    # cannot find unfolded weights as .h5 files. Try loading .npz files
    fpaths_uw = glob.glob(os.path.join(fpath_result_dir, "weights*.npz"))

if not fpaths_uw:
    print(f"ERROR: cannot find weight files in {fpath_result_dir}")
    sys.exit(4)

fpaths_uw.sort()

run_cfg.update({"unfolded_weights": fpaths_uw})

# if new output directory is provided:
if args.outdir_new:
    if not os.path.isdir(args.outdir_new):
        print("Create new output directory")
        os.makedirs(args.outdir_new)

    run_cfg["outputdir"] = args.outdir_new

if args.no_ibu:
    run_cfg["run_ibu"] = False

# Rerun
print("Rerun plotting")
unfold(**run_cfg)
