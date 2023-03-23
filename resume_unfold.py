#!/usr/bin/env python3
import os
import sys
import util
from unfoldv2 import unfold

# Usage
if len(sys.argv) != 2 :
    print("Usage: ./resume_unfold.py <path_to_output_dir>")
    sys.exit(1)

fpath_result_dir = sys.argv[1]

# check if directory exists
if not os.path.isdir(fpath_result_dir):
    print(f"ERROR: cannot find directory {fpath_result_dir}")
    sys.exit(2)

fpath_args = os.path.join(fpath_result_dir, "arguments.json")
if not os.path.isfile(fpath_args):
    print(f"ERROR: cannot find argument configuration for the previous run: {fpath_args}")
    sys.exit(2)

run_cfg = util.read_dict_from_json(fpath_args)
if not run_cfg:
    print(f"ERROR: fail to load run config from {fpath_args}")
    sys.exit(3)

run_cfg['resume'] = True

print("Resume unfolding")
unfold(**run_cfg)