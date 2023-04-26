"""
Module to handle binned corrections
"""
import os

import histogramming as myhu
import FlattenedHistogram as fh

import logging
logger = logging.getLogger("binnedCorrections")

from ttbarDiffXsRun2.helpers import ttbar_diffXs_run2_params

def compute_binned_corrections(
    observable,
    histograms_dict,
    flow,
    mcw = False
    ):

    if mcw:
        h_resp = histograms_dict[observable][f"h2d_{observable}_response_mcWeight"]
    else:
        h_resp = histograms_dict[observable][f"h2d_{observable}_response"]

    h_reco = histograms_dict[observable][f"h_{observable}_reco"]
    h_truth = histograms_dict[observable][f"h_{observable}_truth"]

    # Acceptance correction
    h_acc = myhu.projectToXaxis(h_resp, flow=flow)
    h_acc.name = "acceptance"
    h_acc = myhu.divide(h_acc, h_reco)

    # Efficiency correction
    # Scale the truth distribution by 1. / branching_ratio
    # https://gitlab.cern.ch/ttbarDiffXs13TeV/ttbarunfold/-/blob/42d07fa2d49bbf699905e05bbb86c6c6b68a8dbf/src/Spectrum.cxx#L645
    h_truth *= 1. / ttbar_diffXs_run2_params['branching_ratio']

    # h_resp_mcw instead?
    h_eff = myhu.projectToYaxis(h_resp, flow=flow)
    h_eff.name = "efficiency"
    h_eff = myhu.divide(h_eff, h_truth)

    return h_acc, h_eff

def compute_binned_corrections_multidim(
    observable,
    histograms_dict,
    flow,
    mcw = False
    ):

    obs_list = observable.split("_vs_")
    if len(obs_list) == 2:
        hname_prefix = "fh2d"
    elif len(obs_list) == 3:
        hname_prefix = "fh3d"
    else:
        raise RuntimeError(f"Observable {observable} is neither 2D nor 3D")

    if mcw:
        h_resp = histograms_dict[observable][f"{hname_prefix}_{observable}_response_mcWeight"]
    else:
        h_resp = histograms_dict[observable][f"{hname_prefix}_{observable}_response"]

    h_reco = histograms_dict[observable][f"{hname_prefix}_{observable}_reco"]
    h_truth = histograms_dict[observable][f"{hname_prefix}_{observable}_truth"]

    # Acceptance correction
    h_acc_flat = myhu.projectToXaxis(h_resp, flow=flow)

    h_acc = h_reco.copy()
    h_acc.reset()
    h_acc.fromFlat(h_acc_flat)

    h_acc.divide(h_reco)

    # Efficiency correction

    # Scale the truth distribution by 1. / branching_ratio
    # https://gitlab.cern.ch/ttbarDiffXs13TeV/ttbarunfold/-/blob/42d07fa2d49bbf699905e05bbb86c6c6b68a8dbf/src/Spectrum.cxx#L645
    h_truth.scale(1. / ttbar_diffXs_run2_params['branching_ratio'])

    h_eff_flat = myhu.projectToYaxis(h_resp,flow=flow)

    h_eff = h_truth.copy()
    h_eff.reset()
    h_eff.fromFlat(h_eff_flat)

    h_eff.divide(h_truth)

    return h_acc, h_eff

def collect_histograms(
    histograms_dir, # str, top directory to collect histograms for computing corrections
    observables=[], # list of str, names of observables to compute corrections
    histogram_suffix = "_histograms.root",
    output_name = None
    ):

    if not os.path.isdir(histograms_dir):
        logger.error(f"Cannot access directory {histograms_dir}")
        return {}

    logger.info(f"Collect histogram files from {histograms_dir}")
    fpaths_histogram = []
    for r, d, files in os.walk(histograms_dir):
        for fname in files:
            if fname.endswith(histogram_suffix):
                logger.debug(f" {os.path.join(r,fname)}")
                fpaths_histogram.append(os.path.join(r,fname))

    if not fpaths_histogram:
        logger.error(f"Found no histogram file in {histograms_dir}")

    histograms_d = {}

    logger.info("Read histograms from files")
    for fpath in fpaths_histogram:
        hists_file_d = myhu.read_histograms_dict_from_file(fpath)

        if not observables:
            observables = list(hists_file_d.keys())

        for ob in observables:

            if not ob in hists_file_d:
                logger.error(f"Cannot find histograms for {ob} in {fpath}")
                continue

            if not ob in histograms_d:
                histograms_d[ob] = {}

            for hname in hists_file_d[ob]:
                if hname.startswith("Acceptance_") or hname.startswith("Efficiency_"):
                    continue

                if not hname in histograms_d[ob]:
                    histograms_d[ob][hname] = hists_file_d[ob][hname]
                else:
                    histograms_d[ob][hname] += hists_file_d[ob][hname]

    if output_name:
        logger.info(f"Write histograms to file {output_name}")
        myhu.write_histograms_dict_to_file(histograms_d, output_name)

    return histograms_d

def binned_corrections(
    histograms_dict, # dict, collection of histograms from collect_histograms()
    observables=[], # list of str, names of observables to compute corrections
    flow = True, # bool, if True, include underflow and overflow bins
    mc_weight = False, # bool, if True, use response with mc weight
    output_name = None
    ):

    logger.info("Compute corrections")

    corrections_d = {}

    if not observables:
        observables = list(histograms_dict.keys())

    for obs in observables:
        logger.info(obs)

        obs_list = obs.split("_vs_")

        corrections_d[obs] = {}

        if len(obs_list) == 1:
            h_acc, h_eff = compute_binned_corrections(obs, histograms_dict, flow=flow, mcw=mc_weight)
        else:
            h_acc, h_eff = compute_binned_corrections_multidim(obs, histograms_dict, flow=flow, mcw=mc_weight)

        corrections_d[obs]['acceptance'] = h_acc
        corrections_d[obs]['efficiency'] = h_eff

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
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Name of the output file for binned corrections")
    parser.add_argument('--observables', nargs='+', type=str,
                        help="List of observables. Use all that are available in the histograms if not specified")
    parser.add_argument('--no-flow', action='store_true',
                        help="If True, exclude underflow and overflow bins")
    parser.add_argument('--mc-weight', action='store_true',
                        help="If True, use the response with mc weights for computing binned corrections")
    parser.add_argument('--histogram-outname', type=str,
                        help="If specified, save the collected histograms to 'histogram_outname'")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set the logging level to DEBUG.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if os.path.isdir(args.fpath_histograms):
        hists_d = collect_histograms(args.fpath_histograms, args.observables)
    else:
        hists_d = myhu.read_histograms_dict_from_file(args.fpath_histograms)

    if args.histogram_outname:
        logger.info(f"Write histograms to file {args.histogram_outname}")
        myhu.write_histograms_dict_to_file(hists_d, args.histogram_outname)

    binned_corrections(
        hists_d,
        observables=args.observables,
        flow = not args.no_flow,
        mc_weight = args.mc_weight,
        output_name=args.output
        )
