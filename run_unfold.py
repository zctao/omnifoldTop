#!/usr/bin/env python3
import os
import sys
import itertools
from util import read_dict_from_json, reportMemUsage
from unfoldv2 import getArgsParser, unfold
from modelUtils import reportGPUMemUsage

import logging
logger = logging.getLogger("run_unfold")

def run_unfold(fpath_run_config):
    if not os.path.isfile(fpath_run_config):
        raise FileNotFoundError(f"Cannot find run config: {fpath_run_config}")

    run_cfgs = read_dict_from_json(fpath_run_config)
    if not run_cfgs:
        raise RuntimeError(f"Failed to load run config from {fpath_run_config}")

    # run_cfgs could be a dictionary or a list of dictionaries
    if not isinstance(run_cfgs, list):
        run_cfgs = [run_cfgs]

    # loop over the list of dict objects
    for run_cfg_d in run_cfgs:

        # If there is a entry: "skip": true, skip this one
        if run_cfg_d.pop("skip", None):
            continue

        # get the default argument dictionary from getArgsParser
        default_args = getArgsParser([])
        default_args = vars(default_args)

        # Loop over run_cfg_d and replace the default argument values
        scan_args = list()
        for k, v in run_cfg_d.items():
            if isinstance(v, dict):
                # We are going to run unfolding with different values of this
                # argument handle it later
                scan_args.append(k)
            elif k in default_args:
                default_args[k] = v
            else:
                raise ValueError(f"{k} is not a known argument")

        # for arguments with labels and multiple values
        labels = []
        for k in scan_args:
            assert(isinstance(run_cfg_d[k], dict))
            labels.append( list(run_cfg_d[k].keys()) )

        combinations = list(itertools.product(*labels))

        # for each argument combination
        for comb in combinations:
            run_args = default_args.copy()

            assert( len(comb) == len(scan_args) )
            for k, l in zip(scan_args, comb):
                run_args[k] = run_cfg_d[k][l]

            # update output directory based on the labels
            run_args['outputdir'] = os.path.join(run_args['outputdir'], *comb)

            print("Run unfolding")
            #print(run_args)
            unfold(**run_args)

if __name__ == "__main__":

    # Usage
    if len(sys.argv) != 2:
        print("Usage: ./run_unfold.py <run_config.json>")
        sys.exit(0)

    fpath_run_config = sys.argv[1]

    try:
        run_unfold(fpath_run_config)
    except Exception as ex:
        logger.setLevel(logging.DEBUG)
        reportMemUsage(logger)
        reportGPUMemUsage(logger)
        logger.error(f"Unfold failed: {ex}")
        sys.exit(1)