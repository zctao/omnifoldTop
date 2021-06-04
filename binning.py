#!/usr/bin/env python3
import os
import numpy as np
import time
import tracemalloc

from resolution import resolution

from plotting import plot_histograms1d
from util import getLogger, load_dataset, get_variable_arr
from util import read_dict_from_json, write_dict_to_json
from util import make_hist

logger = getLogger('Binning')

observable_dict = read_dict_from_json('configs/observables/default.json')

def binning(**parsed_args):

    tracemalloc.start()

    logger.info("Variables to bin: {}".format(', '.join(parsed_args['variables'])))

    # read input data
    logger.info('Reading dataset {}'.format(parsed_args['inputs']))
    t_data_start = time.time()

    ntuple = load_dataset(parsed_args['inputs'], weight_columns=[parsed_args['weight'], parsed_args['weight_mc']])

    t_data_end = time.time()
    logger.debug("Reading dataset took {:.2f} seconds".format(t_data_end-t_data_start))
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug("Current memory usage: {:.1f} MB; Peak usage: {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    logger.info("Total number of events: {}".format(len(ntuple)))
    logger.debug("Total size of the array: {:.3f} MB".format(ntuple.nbytes*2**-20))

    # event weights
    weights_reco = get_variable_arr(ntuple, parsed_args['weight'])
    weights_truth = get_variable_arr(ntuple, parsed_args['weight_mc'])
    logger.debug("Total reconstructed event weights: {:.3f}".format(weights_reco.sum()))
    logger.debug("Total MC truth weights: {:.3f}".format(weights_truth.sum()))

    fname_bins = os.path.join(parsed_args['outdir'], 'bins.json')

    # for each variable
    for varname in parsed_args['variables']:
        logger.info('Compute bins for {}'.format(varname))

        varname_det = observable_dict[varname]['branch_det']
        varname_mc = observable_dict[varname]['branch_mc']

        var_reco_arr = get_variable_arr(ntuple, varname_det)
        var_truth_arr = get_variable_arr(ntuple, varname_mc)

        bins = compute_bins(varname, var_reco_arr, var_truth_arr,
                            weights_reco, weights_truth,
                            parsed_args['nfinebins'],
                            do_plot=parsed_args['plot'],
                            outdir=parsed_args['outdir'])

        # write bins to output file
        if os.path.isfile(fname_bins): # the file already exists
            # get the binning dictionary in the file
            binning_dict = read_dict_from_json(fname_bins)
        else: # create a new dictionary to write to the output
            binning_dict = {}

        binning_dict[varname] = bins.tolist()
        write_dict_to_json(binning_dict, fname_bins)

    logger.info("Updated binning configuration in {}".format(fname_bins))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug("Current memory usage: {:.1f} MB; Peak usage: {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))

    tracemalloc.stop()

###################################
def compute_bins(varname, data_reco_arr, data_truth_arr, weights_reco, weights_truth, nfinebins, do_plot=False, outdir='.'):
    # variable range
    if observable_dict[varname].get('xlim') is None:
        xlim_low = min(data_reco_arr.min(), data_truth_arr.min())
        xlim_high = max(data_reco_arr.max(), data_truth_arr.max())
    else:
        xlim_low = observable_dict[varname]['xlim'][0]
        xlim_high = observable_dict[varname]['xlim'][1]

    # start with a histogram with finer bin width
    fineBins = np.linspace(xlim_low, xlim_high, nfinebins+1)

    # resolution as a function of truth-level variable value
    figname_resol = os.path.join(outdir,'resolution_'+varname) if do_plot else None
    f_resol_var = resolution(data_reco_arr, data_truth_arr,
                             weights_reco*weights_truth,
                             fineBins, fineBins,
                             figname=figname_resol,
                             xlabel=observable_dict[varname].get('xlabel'))

    # merge bins from left to right based on variable resolutions
    # as well as statistical uncertainties in reco bins

    hist_fine, hist_fine_err = make_hist(data_reco_arr, weights=weights_reco, bins=fineBins)

    delta = observable_dict[varname].get('delta_res', 1.)
    max_bin_err = observable_dict[varname].get('max_bin_error', 0.05)

    resBins = merge_bins(fineBins, f_resol_var, hist_fine, hist_fine_err, delta, max_bin_err)

    if do_plot:
        figname_var = os.path.join(outdir,'{}_binning'.format(varname))
        xlabel_var = observable_dict[varname].get('xlabel')
        hist_reco, hist_reco_err = make_hist(data_reco_arr, weights=weights_reco, bins=resBins)
        hist_truth, hist_truth_err = make_hist(data_truth_arr, weights=weights_truth, bins=resBins)
        plot_histograms1d(figname_var, resBins, [hist_reco, hist_truth], [hist_reco_err, hist_truth_err], labels=['Reco', 'Truth'], xlabel=xlabel_var,plottypes=['g','h'], marker='+')

    return resBins

def merge_bins(finebins, f_resolution, hist, hist_err, delta, max_bin_err):
    newbins = [finebins[0]]
    nsum = 0.
    w2sum = 0.
    for ibin in range(len(finebins)-1):
        # current bin edges
        xlow_cur = newbins[-1]
        xhigh_cur = finebins[ibin+1]

        xmid_cur = (xlow_cur + xhigh_cur) / 2
        binwidth_cur = xhigh_cur - xlow_cur

        # resolution requirement
        resol_cur = f_resolution(xmid_cur)
        width_good = (resol_cur > 0) and (binwidth_cur/2 > delta*resol_cur)

        # bin error requirement
        w2sum += hist_err[ibin]**2
        nsum += hist[ibin]
        if nsum > 0:
            rel_err = np.sqrt(w2sum)/nsum
            error_good = rel_err < max_bin_err
        else:
            error_good = False

        if width_good and error_good:
            # add the current bin edge
            newbins.append(xhigh_cur)
            nsum = 0.
            w2sum = 0.
        elif ibin == len(finebins)-2: # in case of the last bin
            # merge what is left with the previous bin
            newbins[-1] = xhigh_cur
            # or make a new bin?
            # newbins.append(xhigh_cur)

    return np.asarray(newbins)

def get_bins(varname, fname_bins='bins.json'):
    if os.path.isfile(fname_bins):
        # read bins from the config file
        binning_dict = read_dict_from_json(fname_bins)
        if varname in binning_dict:
            if isinstance(binning_dict[varname], list):
                return np.asarray(binning_dict[varname])
            elif isinstance(binning_dict[varname], dict):
                # equal bins
                return np.linspace(binning_dict[varname]['xmin'], binning_dict[varname]['xmax'], binning_dict[varname]['nbins']+1)
        else:
            logger.debug("  No binning information found is {} for {}".format(fname_bins, varname))
    else:
        logger.warn("  No binning config file {}".format(fname_bins))

    # if the binning file does not exist or no binning info for this variable is in the dictionary
    return None

if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser()

    parser.add_argument('variables', nargs='+', choices=observable_dict.keys(),
                        help="variables to bin")
    parser.add_argument('-i', '--inputs', required=True, nargs='+', type=str,
                        help="Input files")
    parser.add_argument('-o', '--outdir', default='configs',
                        help="Output file name")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="plot distributions with new bins if true")
    parser.add_argument('--weight', default='w',
                        help="name of event weight")
    parser.add_argument('--weight-mc', dest='weight_mc', default='wTruth',
                        help="name of MC weight")
    parser.add_argument('--nfinebins', default=100, type=int,
                        help="number of fine bins to start with")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbosity level")

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose > 0 else logging.INFO)

    if not os.path.isdir(args.outdir):
        logger.info("Create directory: {}".format(args.outdir))
        os.makedirs(args.outdir)

    binning(**vars(args))
