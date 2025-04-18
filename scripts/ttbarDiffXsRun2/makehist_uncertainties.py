#!/usr/bin/env python3
import os
import json
import subprocess

import util
from ttbarDiffXsRun2.systematics import uncertainty_groups
from slurm.generate_slurm_hist_jobs import generate_slurm_hist_jobs

import logging
logger = logging.getLogger("run_uncertainties_hist")

def match_keywords(syst_name, keywords_match=[]):
    for kw in keywords_match:
        if kw in syst_name:
            return True
    return False

def generate_jobs(
    job_name,
    syst_topdir,
    observables,
    status_d,
    job_dir = ".",
    keywords_match = [],
    rerun = False
    ):

    systs_gen = []

    for systname in status_d:
        if not match_keywords(systname, keywords_match):
            #logger.debug(f"None of the keywords in {keywords_match} matches {systname}. Skip.")
            continue

        # check status
        if status_d.get(systname) == 'histogram':
            if not rerun:
                logger.debug(f"Histograms for {systname} have already been generated before. Skip.")
                continue
        elif status_d.get(systname) != "unfold":
            logger.warning(f"Unfolding has not been done for {systname} yet! Skip.")
            continue

        # add to the list for generating job files
        systs_gen.append(systname)

    # generate slurm job files
    hist_args_list = [
        f"--observables {' '.join(observables)}",
        "--observable-config ${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json",
        "--binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json",
        "--include-ibu"
    ]

    result_dirs = [os.path.join(syst_topdir, systname) for systname in systs_gen]

    jobfilename = generate_slurm_hist_jobs(
        job_name,
        result_dirs,
        sample_dir = "./inputs",
        tarball_dir = os.path.expandvars("${DATA_DIR}/ntuplerTT/latest/tarballs"),
        submitdir = job_dir,
        histograms_args = " ".join(hist_args_list)
    )

    # update status
    for systname in systs_gen:
        status_d[systname] = "histogram"

    logger.info(f"Generated slurm job file: {jobfilename}")
    logger.debug(f"  Included systematics: {systs_gen}")

    return jobfilename

def makehist_uncertainties(
    job_name,
    syst_topdir,
    systematics_groups = [], # list of str, systematic groups
    systematics_keywords = [], # list of str, keywords for selecting a subset of systematic uncertainties
    observables = ["mtt", "ptt", "th_pt", "tl_pt", "ytt", "th_y", "tl_y", "ptt_vs_mtt", "th_pt_vs_mtt", "ytt_abs_vs_mtt", "ptt_vs_ytt_abs", "mtt_vs_ytt_abs", "mtt_vs_ptt_vs_ytt_abs", "mtt_vs_th_pt_vs_th_y_abs", "mtt_vs_th_pt_vs_ytt_abs", "mtt_vs_th_y_abs_vs_ytt_abs"],
    job_dir = '.',
    rerun = False,
    submit = False,
    verbose = False
    ):

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # status files
    fpath_job_status = os.path.join(syst_topdir, "status", "jobs.json")
    jstatus_unfold = util.read_dict_from_json(fpath_job_status)

    fpath_hist_status = os.path.join(syst_topdir, "status", "jobs_hists.json")
    if os.path.isfile(fpath_hist_status):
        # read status from the existing file if exist
        logger.debug(f"Load histogram job status from {fpath_hist_status}")
        with open(fpath_hist_status, "r") as fhist:
            jstatus_hist = json.load(fhist)
    else:
        # create a new dict
        jstatus_hist = dict()

    # merge the two
    jstatus = jstatus_unfold.copy()
    for syst in jstatus_hist:
        if jstatus_hist[syst] == 'histogram':
            jstatus[syst] = jstatus_hist[syst]

    # keywords to pick systematics
    keywords_syst = []
    keywords_syst += systematics_keywords
    for group in systematics_groups:
        keywords_syst += uncertainty_groups[group]["filters"]

    fname_job = generate_jobs(
        job_name,
        syst_topdir,
        observables = observables,
        status_d = jstatus,
        job_dir = job_dir,
        keywords_match = keywords_syst,
        rerun = rerun
    )

    if submit:
        logger.info(f"Submit slurm job: {fname_job}")
        subprocess.run(["sbatch", f"{fname_job}"], check=True)

    # update the status file
    with open(fpath_hist_status, "w") as jstatus_hist_new:
        json.dump(jstatus, jstatus_hist_new, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("job_name", type=str,
                        help="Slurm job name")
    parser.add_argument("-s", "--syst-topdir", type=str, required=True,
                        help="Top directory of the unfolding results for systematic uncertainty variations")
    parser.add_argument("-g", "--systematics-groups", nargs="*", type=str, default=[],
                        help="Groups of systematic uncertainties")
    parser.add_argument("-k", "--systematics-keywords", nargs="*", type=str, default=[],
                        help="Additional keywords for selecting a subset of systematic uncertainties")
    parser.add_argument("--observables", type=str, nargs='*',
                        help="List of observables to evaluate bin uncertainties")
    parser.add_argument("-j", "--job-dir", type=str, default=".",
                        help="Job submission directory")
    parser.add_argument("--rerun", action="store_true",
                        help="If True, rerun job generation even if they have already been generated.")
    parser.add_argument("-b", "--submit", action="store_true",
                        help="If True, submit the slurm batch job after generating the job file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="If True, set logging level to debug, else info")

    args = parser.parse_args()

    if not os.path.isdir(args.job_dir):
        logger.info(f"Create job directory {args.job_dir}")
        os.makedirs(args.job_dir)

    util.configRootLogger()

    makehist_uncertainties(**vars(args))