#!/usr/bin/env python3
import os
import logging
import numpy as np

import util
from OmniFoldTTbar import OmniFoldTTbar
import histogramming as myhu
from plotter import plot_uncertainties

def compute_uncertainties(
        variable,
        bins,
        uf_nominal,
        uf_syst_up,
        uf_syst_down,
        iteration=-1
    ):

    # unfolding uncertainty from the nominal distribution
    h_uf_nominal = uf_nominal.get_unfolded_distribution(variable, bins, iteration=iteration)[0]
    hval_nominal, herr_of = myhu.get_values_and_errors(h_uf_nominal)
    relerr_of = herr_of / hval_nominal

    # systematic uncertainty from variations
    # histograms from resamples
    h_uf_nominal_rs = uf_nominal.get_unfolded_hists_resamples(variable, bins, iteration=iteration)
    h_uf_syst_up_rs = uf_syst_up.get_unfolded_hists_resamples(variable, bins, iteration=iteration)
    h_uf_syst_down_rs = uf_syst_down.get_unfolded_hists_resamples(variable, bins, iteration=iteration)

    # in case there is no resampled weights
    if not h_uf_nominal_rs:
        h_uf_nominal_rs = [h_uf_nominal]
    if not h_uf_syst_up_rs:
        h_uf_syst_up_rs = [uf_syst_up.get_unfolded_distribution(variable, bins, iteration=iteration)[0]]
    if not h_uf_syst_down_rs:
        h_uf_syst_down_rs = [uf_syst_down.get_unfolded_distribution(variable, bins, iteration=iteration)[0]]

    # list of errors
    relerr_syst_up_rs = []
    relerr_syst_down_rs = []
    for h_nom, h_up, h_down in zip(h_uf_nominal_rs, h_uf_syst_up_rs, h_uf_syst_down_rs):
        relerr_syst_up_rs.append( h_up.values() / h_nom.values() - 1. )
        relerr_syst_down_rs.append( h_down.values() / h_nom.values() - 1. )

    # mean
    relerr_syst_up = np.mean(relerr_syst_up_rs, axis=0)
    relerr_syst_down = np.mean(relerr_syst_down_rs, axis=0)

    return relerr_of, relerr_syst_up, relerr_syst_down

def evaluate_systematics(**parsed_args):
    logger = logging.getLogger("EvalSyst")

    # Print arguments to logger
    for argkey, argvalue in sorted(parsed_args.items()):
        if argvalue is None:
            continue
        logger.info(f"Argument {argkey}: {argvalue}")

    # Get run arguments from unfolding output directories
    fname_args_nominal = os.path.join(parsed_args['nominal'], 'arguments.json')
    logger.debug(f"Reading arguments from {fname_args_nominal}")
    args_nom_dict = util.read_dict_from_json(fname_args_nominal)

    ####
    # variables
    observable_config = args_nom_dict['observable_config']
    logger.debug(f"Get observable config from {observable_config}")
    observable_dict = util.read_dict_from_json(observable_config)
    assert(observable_dict)

    if not parsed_args['observables']:
        # if the list of observables is not specified, get it from args_nom_dict
        observables = args_nom_dict['observables'] + args_nom_dict['observables_extra']
    else:
        observables = parsed_args['observables']

    varnames_reco = [ observable_dict[key]['branch_det'] for key in observables ]
    varnames_truth = [ observable_dict[key]['branch_mc'] for key in observables ]

    # Nominal
    logger.info("Unfolder - nominal")
    unfolder_nom = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        filepaths_obs = args_nom_dict['data'],
        filepaths_sig = args_nom_dict['signal'],
        filepaths_bkg = args_nom_dict['background'],
        normalize_to_data = args_nom_dict['normalize'],
        dummy_value = args_nom_dict['dummy_value'],
        weight_type = args_nom_dict['weight_type']
    )

    # load weights
    fnames_uw_nom = [os.path.join(args_nom_dict['outputdir'], "weights.npz")]
    if args_nom_dict['error_type'] != 'sumw2':
        fnames_uw_nom.append(os.path.join(args_nom_dict['outputdir'], f"weights_resample{args_nom_dict['nresamples']}.npz"))

    logger.info(f"Load weights from {fnames_uw_nom}")
    unfolder_nom.load(fnames_uw_nom)

    # Systematic uncertainty up variation
    logger.info("Unfolder - up")

    fname_args_up = os.path.join(parsed_args['syst_up'], 'arguments.json')
    logger.debug(f"Reading arguments from {fname_args_up}")
    args_up_dict = util.read_dict_from_json(fname_args_up)

    unfolder_up = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        filepaths_obs = args_up_dict['data'],
        filepaths_sig = args_up_dict['signal'],
        filepaths_bkg = args_up_dict['background'],
        normalize_to_data = args_up_dict['normalize'],
        dummy_value = args_up_dict['dummy_value'],
        weight_type = args_up_dict['weight_type']
    )

    # load weights
    fnames_uw_up = [os.path.join(args_up_dict['outputdir'], "weights.npz")]
    if args_up_dict['error_type'] != 'sumw2':
        fnames_uw_up.append(os.path.join(args_up_dict['outputdir'], f"weights_resample{args_up_dict['nresamples']}.npz"))

    logger.info(f"Load weights from {fnames_uw_up}")
    unfolder_up.load(fnames_uw_up)

    # Systematic uncertainty down variation
    logger.info("Unfolder - down")

    fname_args_down = os.path.join(parsed_args['syst_down'], 'arguments.json')
    logger.debug(f"Reading arguments from {fname_args_down}")
    args_down_dict = util.read_dict_from_json(fname_args_down)

    unfolder_down = OmniFoldTTbar(
        varnames_reco,
        varnames_truth,
        filepaths_obs = args_down_dict['data'],
        filepaths_sig = args_down_dict['signal'],
        filepaths_bkg = args_down_dict['background'],
        normalize_to_data = args_down_dict['normalize'],
        dummy_value = args_down_dict['dummy_value'],
        weight_type = args_down_dict['weight_type']
    )

    # load weights
    fnames_uw_down = [os.path.join(args_down_dict['outputdir'], "weights.npz")]
    if args_down_dict['error_type'] != 'sumw2':
        fnames_uw_down.append(os.path.join(args_down_dict['outputdir'], f"weights_resample{args_down_dict['nresamples']}.npz"))

    logger.info(f"Load weights from {fnames_uw_down}")
    unfolder_down.load(fnames_uw_down)

    ####
    # plot uncertainties
    if not os.path.isdir(parsed_args['outputdir']):
        logger.info(f"Create output directory {parsed_args['outputdir']}")
        os.makedirs(parsed_args['outputdir'])

    # binning
    bin_config = parsed_args['binning_config'] if parsed_args['binning_config'] else args_nom_dict['binning_config']

    for ob in observables:
        logger.info(f"Plot {ob}")
        bin_edges = util.get_bins(ob, bin_config)

        varname_truth = observable_dict[ob]['branch_mc']

        relerr_uf, relerr_up, relerr_down = compute_uncertainties(
            varname_truth, bin_edges,
            unfolder_nom, unfolder_up, unfolder_down,
            iteration = parsed_args['iteration'])

        plot_uncertainties(
            figname = os.path.join(parsed_args['outputdir'], f'relerr_{ob}'),
            bins = bin_edges,
            uncertainties = [relerr_uf, (relerr_up, relerr_down)],
            draw_options = [
                {'label':'OmniFold', 'edgecolor':'tab:red', 'facecolor':'none'},
                {'label':parsed_args['systematics'], 'hatch':'///', 'edgecolor':'tab:blue', 'facecolor':'none'}
                ],
            xlabel = observable_dict[ob]['xlabel'],
            ylabel = 'Uncertainty'
            )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('systematics', type=str,
                        help='Name of the systematic uncertainty')
    parser.add_argument('--nominal', required=True, type=str,
                        help="Directory to load nominal unfolded weights")
    parser.add_argument('--syst-up', required=True, type=str,
                        help="Directory to load unfolded weights for upward variation of the systematic uncertainty")
    parser.add_argument('--syst-down', required=True, type=str,
                        help="Directory to load unfolded weights for downward variation of the systematic uncertainty")
    parser.add_argument('-o', '--outputdir', type=str, default='.',
                        help="Output directory")
    parser.add_argument('--observables', nargs='+',
                        help="List of observables to use in training.")
    parser.add_argument('--binning-config', dest='binning_config', type=str,
                        help="Binning config file for variables")
    parser.add_argument('--iteration', type=int, default=-1,
                        help="Use unfolded weights at the specified iteration")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='If True, set logging level to DEBUG')

    args = parser.parse_args()

    # logging
    logfile = os.path.join(args.outputdir, 'log.txt')
    util.configRootLogger(filename=logfile)
    logger = logging.getLogger("EvalSyst")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # output directory
    if not os.path.isdir(args.outputdir):
        logger.info(f"Create output directory {args.outputdir}")
        os.makedirs(args.outputdir)

    # Evaluate systematics
    evaluate_systematics(**vars(args))
