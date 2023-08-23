"""
Module to handle binned corrections
"""
import os
import time

import histogramming as myhu
import FlattenedHistogram as fh
import util
from datahandler_root import DataHandlerROOT

import logging
logger = logging.getLogger("binnedCorrections")

from ttbarDiffXsRun2.helpers import ttbar_diffXs_run2_params

# Acceptance correction
def compute_binned_acceptance(hist_response, hist_reco, flow):
    hist_acc = myhu.projectToXaxis(hist_response, flow=flow)
    hist_acc.name = "acceptance"
    hist_acc = myhu.divide(hist_acc, hist_reco)
    return hist_acc

def compute_binned_acceptance_multidim(fhist_response, fhist_reco, flow):
    fhist_acc = fhist_response.projectToReco(flow=flow)
    fhist_acc.name = "acceptance"
    fhist_acc.divide(fhist_reco)

    return fhist_acc

# Efficiency correction
def compute_binned_efficiency(hist_response, hist_truth, flow):
    # Scale the truth distribution by 1. / branching_ratio
    # https://gitlab.cern.ch/ttbarDiffXs13TeV/ttbarunfold/-/blob/42d07fa2d49bbf699905e05bbb86c6c6b68a8dbf/src/Spectrum.cxx#L645
    hist_truth = hist_truth * ( 1./ttbar_diffXs_run2_params['branching_ratio'] )

    hist_eff = myhu.projectToYaxis(hist_response, flow=flow)
    hist_eff.name = "efficiency"
    hist_eff = myhu.divide(hist_eff, hist_truth)

    return hist_eff

def compute_binned_efficiency_multidim(fhist_response, fhist_truth, flow):
    # Scale the truth distribution by 1. / branching_ratio
    fhist_truth.scale( 1./ttbar_diffXs_run2_params['branching_ratio'] )

    fhist_eff = fhist_response.projectToTruth(flow=flow)
    fhist_eff.name = "efficiency"
    fhist_eff.divide(fhist_truth)

    return fhist_eff

def binned_corrections_observable(
    observable,
    histograms_d,
    handle_sim = None,
    obsConfig_d = None,
    flow=True
    ):

    # histograms
    h_reco = histograms_d[observable][f"h_{observable}_reco"]
    h_truth = histograms_d[observable][f"h_{observable}_truth"]

    # response
    if handle_sim is not None:
        # recompute response from data handler
        if not obsConfig_d:
            raise RuntimeError("Cannot load arrays from data handler: no observable configuration dictionary provided")

        resp = handle_sim.get_response(
            obsConfig_d[observable]['branch_det'],
            obsConfig_d[observable]['branch_mc'],
            bins_reco = h_reco.axes[0].edges,
            bins_truth = h_truth.axes[0].edges,
            absoluteValue = '_abs' in observable,
            normalize_truthbins = False
            )
    else:
        # get the response from histograms_d
        resp = histograms_d[observable][f"h2d_{observable}_response"]

    acceptance = compute_binned_acceptance(resp, h_reco, flow=flow)
    efficiency = compute_binned_efficiency(resp, h_truth, flow=flow)

    hists_out_d = {
        "acceptance": acceptance,
        "efficiency": efficiency,
        "response": resp,
        "hreco": h_reco,
        "htruth": h_truth
    }

    return hists_out_d

def binned_corrections_observable_multidim(
    observable,
    histograms_d,
    handle_sim = None,
    obsConfig_d = None,
    flow=True
    ):

    obs_list = observable.split("_vs_")
    if len(obs_list) == 2:
        hname_prefix = "fh2d"
    elif len(obs_list) == 3:
        hname_prefix = "fh3d"
    else:
        raise RuntimeError(f"Observable {observable} is neither 2D nor 3D")

    # histograms
    fh_reco = histograms_d[observable][f"{hname_prefix}_{observable}_reco"]
    fh_truth = histograms_d[observable][f"{hname_prefix}_{observable}_truth"]

    # response
    if handle_sim is not None:
        # recompute response from data handler
        if not obsConfig_d:
            raise RuntimeError("Cannot load arrays from data handler: no observable configuration dictionary provided")

        varnames_reco = [obsConfig_d[obs]['branch_det'] for obs in obs_list]
        varnames_truth = [obsConfig_d[obs]['branch_mc'] for obs in obs_list]

        resp = handle_sim.get_response_flattened(
            varnames_reco,
            varnames_truth,
            fh_reco.get_bins(),
            fh_truth.get_bins(),
            absoluteValues = ["_abs" in obs for obs in obs_list],
            normalize_truthbins = False
        )
    else:
        # get the response from histograms_d
        # FIXME
        resp = histograms_d[observable][f"{hname_prefix}_{observable}_response"]

    acceptance = compute_binned_acceptance_multidim(resp, fh_reco, flow=flow)
    efficiency = compute_binned_efficiency_multidim(resp, fh_truth, flow=flow)

    hists_out_d = {
        "acceptance": acceptance,
        "efficiency": efficiency,
        "response": resp,
        "hreco": fh_reco,
        "htruth": fh_truth
    }

    return hists_out_d

def binned_corrections(
    fpath_histograms,
    fpaths_sample = [],
    observables = [],
    flow = True,
    match_dR = None,
    output_name = None,
    observable_config = 'configs/observables/vars_ttbardiffXs_pseudotop.json'
    ):

    logger.info(f"Read histograms from {fpath_histograms}")
    histograms_dict = myhu.read_histograms_dict_from_file(args.fpath_histograms)

    if output_name:
        outdir = os.path.dirname(output_name)
        if not os.path.isdir(outdir):
            logger.debug(f"Create output directory {outdir}")
            os.makedirs(outdir)

    if not observables:
        observables = list(histograms_dict.keys())

    obsCfg_d = util.read_dict_from_json(observable_config)

    t_dh_start = time.time()

    if fpaths_sample:
        logger.info(f"Load data from {fpaths_sample}")

        varnames_reco = set()
        varnames_truth = set()
        for obs in observables:
            obs_list = obs.split("_vs_")
            for ob in obs_list:
                varnames_reco.add(obsCfg_d[ob]['branch_det'])
                varnames_truth.add(obsCfg_d[ob]['branch_mc'])

        dh_signal = DataHandlerROOT(
            fpaths_sample,
            list(varnames_reco),
            list(varnames_truth),
            treename_reco='reco',
            treename_truth='parton',
            weight_type='nominal',
            match_dR = match_dR
            )

        logger.info(f"Sample loaded")
    else:
        dh_signal = None

    t_dh_done = time.time()

    logger.info("Compute corrections")

    corrections_d = {}

    t_cor_start = time.time()

    for obs in observables:
        logger.info(f" {obs}")

        obs_list = obs.split("_vs_")

        corrections_d[obs] = {}

        if len(obs_list) == 1:
            hists_corr_d = binned_corrections_observable(
                obs, histograms_dict, dh_signal, obsCfg_d, flow=flow
                )
        else:
            hists_corr_d = binned_corrections_observable_multidim(
                obs, histograms_dict, dh_signal, obsCfg_d, flow=flow
                )

        corrections_d[obs].update(hists_corr_d)

    t_cor_done = time.time()
    logger.info("Done")

    logger.debug("Timing:")
    logger.debug(f" Load samples: {(t_dh_done-t_dh_start):.2f} seconds")
    logger.debug(f" Compute corrections: {(t_cor_done-t_cor_start):.2f} seconds")

    if output_name:
        logger.info(f"Write corrections to file {output_name}")
        myhu.write_histograms_dict_to_file(corrections_d, output_name)

    return corrections_d

def apply_acceptance_correction(histogram, h_acc_corr):
    if isinstance(histogram, list):
        return [ apply_acceptance_correction(hh, h_acc_corr) for hh in histogram ]
    elif isinstance(histogram, fh.FlattenedHistogram2D) or isinstance(histogram, fh.FlattenedHistogram3D):
        return histogram.multiply(h_acc_corr)
    else:
        # In case the correction histogram has a different binning
        # Get the correction factors using the histogram's bin centers
        f_acc = myhu.read_histogram_at_locations(histogram.axes[0].centers, h_acc_corr)
        return histogram * f_acc

def apply_efficiency_correction(histogram, h_eff_corr):
    if isinstance(histogram, list):
        return [ apply_efficiency_correction(hh, h_eff_corr) for hh in histogram ]
    elif isinstance(histogram, fh.FlattenedHistogram2D) or isinstance(histogram, fh.FlattenedHistogram3D):
        return histogram.divide(h_eff_corr)
    else:
        # In case the correction histogram has a different binning
        # Get the correction factors using the histogram's bin centers
        f_eff = myhu.read_histogram_at_locations(histogram.axes[0].centers, h_eff_corr)
        return histogram * (1./f_eff)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('fpath_histograms', type=str,
                        help="Filepath of histogram file or top directory to collect histograms for computing binned corrections")
    parser.add_argument('-s', '--samples', type=str, nargs='+',
                        help="List of sample file paths for building response matrices")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Name of the output file for binned corrections")
    parser.add_argument('--observables', nargs='+', type=str,
                        help="List of observables. Use all that are available in the histograms if not specified")
    parser.add_argument('--observable-config', type=str,
                        default="configs/observables/vars_ttbardiffXs_pseudotop.json",
                        help="File path to observable configuration")
    parser.add_argument('--no-flow', action='store_true',
                        help="If True, exclude underflow and overflow bins")
    parser.add_argument('--match-dR', type=float,
                        help="Require dR between the reco and truth tops smaller than the provided value")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set the logging level to DEBUG.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    binned_corrections(
        args.fpath_histograms,
        args.samples,
        observables=args.observables,
        flow = not args.no_flow,
        match_dR = args.match_dR,
        output_name=args.output,
        observable_config = args.observable_config
        )