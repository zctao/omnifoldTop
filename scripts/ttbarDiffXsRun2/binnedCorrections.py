"""
Module to handle binned corrections
"""
import os
import uproot

import util
import histogramming as myhu

from ttbarDiffXsRun2.helpers import get_acceptance_correction, get_efficiency_correction

import logging
logger = logging.getLogger("binnedCorrections")
#logger.setLevel(logging.DEBUG)

def compute_binned_corrections(
    # Name of the observable to compute corrections
    observable,
    # If True, include underflow/overflow bins.
    flow = True,
    # Directory in which to search and collect histograms from ntuplerTT
    search_dir = "/mnt/xrootdg/ztao/NtupleTT/latest/systCRL/ttbar_nominal",
    # Branching ratio to scale the truth distribution
    # Cf. https://gitlab.cern.ch/ttbarDiffXs13TeV/ttbarunfold/-/blob/DM_ljets_resolved/src/Spectrum.cxx#L645
    branching_ratio = 0.438 # for ttbar l+jets
    ):

    logger.info("Collect histogram files")
    fpaths_histogram = []
    for r, dirs, files in os.walk(search_dir):
        for fname in files:
            if fname.endswith("_histograms.root"):
                logger.debug(f" {os.path.join(r,fname)}")
                fpaths_histogram.append(os.path.join(r,fname))

    if len(fpaths_histogram) == 0:
        logger.error(f"Found no histogram file in {search_dir}")
        raise RuntimeError("Fail to compute binned corrections")

    # name of the histogram to read
    hname_resp = f"h2d_{observable}_response"
    hname_resp_mcw = f"h2d_{observable}_response_mcWeight"
    hname_reco = f"h_{observable}_reco"
    hname_truth = f"h_{observable}_truth"

    h_resp = None
    h_resp_mcw = None
    h_reco = None
    h_truth = None

    logger.info("Read histograms from files")
    for fpath in fpaths_histogram:
        with uproot.open(fpath) as fh:
            if h_resp is None:
                h_resp = fh[f"{observable}/{hname_resp}"].to_hist()
            else:
                h_resp += fh[f"{observable}/{hname_resp}"].to_hist()

            if h_resp_mcw is None:
                h_resp_mcw = fh[f"{observable}/{hname_resp_mcw}"].to_hist()
            else:
                h_resp_mcw += fh[f"{observable}/{hname_resp_mcw}"].to_hist()

            if h_reco is None:
                h_reco = fh[f"{observable}/{hname_reco}"].to_hist()
            else:
                h_reco += fh[f"{observable}/{hname_reco}"].to_hist()

            if h_truth is None:
                h_truth = fh[f"{observable}/{hname_truth}"].to_hist()
            else:
                h_truth += fh[f"{observable}/{hname_truth}"].to_hist()

    logger.info("Compute corrections")

    # Acceptance correction
    h_acc = myhu.projectToXaxis(h_resp, flow=flow)
    h_acc.name = "acceptance"
    h_acc = myhu.divide(h_acc, h_reco)

    # Efficiency correction

    # Scale the truth distribution by 1. / branching_ratio
    h_truth *= 1. / branching_ratio

    # h_resp_mcw instead?
    h_eff = myhu.projectToYaxis(h_resp, flow=flow)
    h_eff.name = "efficiency"
    h_eff = myhu.divide(h_eff, h_truth)

    return h_acc, h_eff

def compare_corrections():
    # Compare my binned correction with the ones from DM

    observables = ["th_pt", "mtt"]

    DM_dir = "/mnt/xrootdg/ztao/fromDavide/4j2b_ljets_central/"

    ZT_dir = "/mnt/xrootdg/ztao/NtupleTT/latest/systCRL/ttbar_nominal/"

    for ob in observables:
        print(ob)

        acc_DM = get_acceptance_correction(ob, DM_dir)
        eff_DM = get_efficiency_correction(ob, DM_dir)

        acc_ZT_noflow, eff_ZT_noflow = compute_binned_corrections(ob, flow=False, search_dir=ZT_dir)

        acc_ZT, eff_ZT = compute_binned_corrections(ob, flow=True, search_dir=ZT_dir)

        # Acceptance
        print(f" Acceptance correction ZT/DM: {acc_ZT_noflow.values() / acc_DM.values()}")

        # Efficiency
        print(f" Efficiency correction ZT/DM: {eff_ZT_noflow.values() / eff_DM.values()}")

        # With/without events in underflow/overflow bins
        print(f" Acceptance correction flow/noflow: {acc_ZT.values() / acc_ZT_noflow.values()}")

        print(f" Efficiency correction flow/noflow: {eff_ZT.values() / eff_ZT_noflow.values()}")