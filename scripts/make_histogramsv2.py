import os
import time
import tracemalloc
import numpy as np

import util
import histUtils
import metrics
import FlattenedHistogram as fh
from datahandler_h5 import DataHandlerH5
from OmniFoldTTbar import load_unfolder
import ibu
from plot_histograms import plot_histograms_from_dict

from ttbarDiffXsRun2.helpers import ttbar_diffXs_run2_params

import logging
logger = logging.getLogger("make_histograms")

def compute_reco_distributions(
    variables,
    bins,
    datahandler_sig,
    datahandler_data = None,
    datahandler_bkg = None,
    datahandler_databkg = None,
    absoluteValue = False,
    histograms_obs_d = {},
    ):

    # observed data distribution
    if datahandler_data is not None:
        logger.debug(f" data reco-level distribution")
        histograms_obs_d['reco_data'] = datahandler_data.compute_histogram(
            variables, bins, absoluteValue=absoluteValue
        )

        if datahandler_databkg is not None:
            logger.debug(f" pseudo data background reco-level distribution )")
            histograms_obs_d['reco_data'] += datahandler_databkg.compute_histogram(
                variables, bins, absoluteValue=absoluteValue
            )

    # background simulation
    if datahandler_bkg is not None:
        logger.debug(f" background reco-level distribution")
        histograms_obs_d['reco_bkg'] = datahandler_bkg.compute_histogram(
            variables, bins, absoluteValue=absoluteValue
        )

    # signal simulationi
    logger.debug(f" signal reco-level distrbution")
    histograms_obs_d['reco_sig'] = datahandler_sig.compute_histogram(
        variables, bins, absoluteValue=absoluteValue
    )

    return histograms_obs_d

def compute_reco_distributions_alliters(
    variables,
    bins,
    reco_weights_alliters,
    datahandler_sig,
    absoluteValue = False,
    histograms_obs_d = {},
    ):

    if reco_weights_alliters is None:
        return histograms_obs_d

    weights_sig = datahandler_sig.get_weights(valid_only=True, reco_level=True)
    reco_weights_alliters

    histograms_obs_d['reco_sig_alliters'] = datahandler_sig.compute_histogram(
        variables, bins, absoluteValue=absoluteValue,
        weights = reco_weights_alliters * weights_sig
    )

    return histograms_obs_d

def compute_truth_distributions(
    variables,
    bins,
    datahandler_sig,
    datahandler_data = None, # pseudo data with known MC truth
    datahandler_sig_unmatched = None,
    absoluteValue = False,
    histograms_obs_d = {}
    ):

    # signal prior
    logger.debug(f" signal truth-level distribution (prior)")
    histograms_obs_d['prior'] = datahandler_sig.compute_histogram(
        variables, bins, absoluteValue=absoluteValue, #weights
    )

    if datahandler_data is not None:
        logger.debug(f" pseudo-data truth-level distribution")
        histograms_obs_d['truth'] = datahandler_data.compute_histogram(
            variables, bins, absoluteValue=absoluteValue, #weights
        )

    if datahandler_sig_unmatched is not None:
        logger.debug(f" all generated signal truth-level distribution (generated)")
        histograms_obs_d['generated'] = datahandler_sig.compute_histogram(variables, bins, absoluteValue=absoluteValue) + datahandler_sig_unmatched.compute_histogram(variables, bins, absoluteValue=absoluteValue)

        # scale by 1. / signal_branching_ratio to convert to inclusive ttbar
        # (https://gitlab.cern.ch/ttbarDiffXs13TeV/ttbarunfold/-/blob/42d07fa2d49bbf699905e05bbb86c6c6b68a8dbf/src/Spectrum.cxx#L645)
        histograms_obs_d['generated'] = histograms_obs_d['generated'] * ( 1./ttbar_diffXs_run2_params['branching_ratio'] )

        # also compute the theory differential cross-sections from MC
        # divided by luminosity
        histograms_obs_d['absoluteDiffXs_MC'] = histograms_obs_d['generated'] * (1. / ttbar_diffXs_run2_params['luminosity'])

        if isinstance(histograms_obs_d['absoluteDiffXs_MC'], fh.FlattenedHistogram):
            histograms_obs_d['absoluteDiffXs_MC'].make_density()
        else:
            histograms_obs_d['absoluteDiffXs_MC'] = histograms_obs_d['absoluteDiffXs_MC'] / histUtils.get_hist_widths(histograms_obs_d['absoluteDiffXs_MC'])

        histograms_obs_d['relativeDiffXs_MC'] = histograms_obs_d['generated'].copy()
        if isinstance(histograms_obs_d['relativeDiffXs_MC'], fh.FlattenedHistogram):
            histograms_obs_d['relativeDiffXs_MC'].renormalize(norm=1., density=False)
            histograms_obs_d['relativeDiffXs_MC'].make_density()
        else:
            histUtils.renormalize_hist(histograms_obs_d['relativeDiffXs_MC'], density=False)
            histograms_obs_d['relativeDiffXs_MC'] = histograms_obs_d['relativeDiffXs_MC'] / histUtils.get_hist_widths(histograms_obs_d['relativeDiffXs_MC'])

    return histograms_obs_d

def compute_truth_distributions_noflow(
    variable_reco,
    variable_truth,
    bins_reco,
    bins_truth,
    datahandler_sig,
    datahandler_data = None,
    absoluteValue = False,
    histograms_obs_d = {}
    ):

    if isinstance(bins_reco, dict) and isinstance(bins_truth, dict):
        logger.debug(f" signal truth-level distribution (prior excluding overflow/underflow)")
        h2d_sig = datahandler_sig.get_response_flattened(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValues = absoluteValue,
            normalize_truthbins = False
        )
        histograms_obs_d['prior_noflow'] = h2d_sig.projectToTruth(flow=False)

        if datahandler_data is not None:
            logger.debug(f" pseudo-data truth-level distribution (excluding overflow/underflow)")
            # pseudo-data MC truth
            h2d_data = datahandler_data.get_response_flattened(
                variable_reco, variable_truth,
                bins_reco, bins_truth,
                absoluteValues = absoluteValue,
                normalize_truthbins = False
            )
            histograms_obs_d['truth_noflow'] = h2d_data.projectToTruth(flow=False)
    else:
        logger.debug(f" signal truth-level distribution (prior excluding overflow/underflow)")
        h2d_sig = datahandler_sig.get_response(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValue = absoluteValue,
            normalize_truthbins = False
        )

        histograms_obs_d['prior_noflow'] = histUtils.projectToYaxis(h2d_sig, flow=False)

        if datahandler_data is not None:
            logger.debug(f" pseudo-data truth-level distribution (excluding overflow/underflow)")
            # pseudo-data MC truth
            h2d_data = datahandler_data.get_response(
                variable_reco, variable_truth,
                bins_reco, bins_truth,
                absoluteValue = absoluteValue,
                normalize_truthbins = False
            )

            histograms_obs_d['truth_noflow'] = histUtils.projectToYaxis(h2d_data, flow=False)

    return histograms_obs_d

def compute_response(
    variables_reco,
    variables_truth,
    bins_reco,
    bins_truth,
    datahandler,
    absoluteValue = False,
    histograms_obs_d = {}
    ):
    logger.debug(f" response")

    if isinstance(bins_reco, dict) and isinstance(bins_truth, dict):
        # fh.FlattenedResponse
        histograms_obs_d['response'] = datahandler.get_response_flattened(
            variables_reco, variables_truth,
            bins_reco, bins_truth,
            absoluteValues = absoluteValue,
            normalize_truthbins=True
        )
    else:
        histograms_obs_d['response'] = datahandler.get_response(
            variables_reco, variables_truth,
            bins_reco, bins_truth,
            absoluteValue = absoluteValue,
            normalize_truthbins = True
        )

    return histograms_obs_d

def compute_binned_acceptance(
    variable_reco,
    bins_reco,
    datahandler,
    absoluteValue = False,
    histograms_obs_d = {}
    ):
    # denominator: all valid events at the reco level
    if not 'reco_sig' in histograms_obs_d:
        compute_reco_distributions(
            variable_reco, bins_reco,
            datahandler_sig = datahandler,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

    h_acc_denom = histograms_obs_d['reco_sig']

    # numerator: truth-matched events at the reco level
    ismatched = datahandler.pass_reco & datahandler.pass_truth
    h_acc_numer = datahandler.compute_histogram(variable_reco, bins_reco, absoluteValue=absoluteValue, extra_cuts=ismatched)

    if isinstance(h_acc_denom, fh.FlattenedHistogram):
        histograms_obs_d['acceptance'] = h_acc_numer
        histograms_obs_d['acceptance'].divide(h_acc_denom)
    else:
        histograms_obs_d['acceptance'] = histUtils.divide(h_acc_numer, h_acc_denom)

    return histograms_obs_d

def compute_binned_acceptance_noflow(
    variable_reco,
    variable_truth,
    bins_reco,
    bins_truth,
    datahandler,
    absoluteValue = False,
    histograms_obs_d = {}
    ):
    # denominator: all valid events at the reco level
    if not 'reco_sig' in histograms_obs_d:
        compute_reco_distributions(
            variable_reco, bins_reco,
            datahandler_sig = datahandler,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

    h_acc_denom = histograms_obs_d['reco_sig']

    # numerator: truth-matched events at the reco level
    # for debug/comparison: exclude overflow and underflow events
    if isinstance(h_acc_denom, fh.FlattenedHistogram):
        h2d = datahandler.get_response_flattened(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValues = absoluteValue,
            normalize_truthbins = False
        )

        h_acc_numer_noflow = h2d.projectToReco(flow=False)
        histograms_obs_d['acceptance_noflow'] = h_acc_numer_noflow
        histograms_obs_d['acceptance_noflow'].divide(h_acc_denom)
    else:
        h2d = datahandler.get_response(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValue = absoluteValue,
            normalize_truthbins = False
        )

        h_acc_numer_noflow = histUtils.projectToXaxis(h2d, flow=False)
        histograms_obs_d['acceptance_noflow'] = histUtils.divide(h_acc_numer_noflow, h_acc_denom)

    return histograms_obs_d

def compute_binned_efficiency(
    variable_truth,
    bins_truth,
    datahandler,
    datahandler_unmatched,
    absoluteValue = False,
    histograms_obs_d = {}
    ):
    if datahandler_unmatched is None:
        logger.warning("Cannot compute binned efficiency: unmatched data are not provided")
        return histograms_obs_d

    # denominator: all valid truth-level events, matched or unmatched
    if not 'generated' in histograms_obs_d:
        # compute truth-level distributions
        compute_truth_distributions(
            variable_truth, bins_truth,
            datahandler_sig = datahandler,
            datahandler_sig_unmatched = datahandler_unmatched,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

    h_eff_denom = histograms_obs_d['generated']

    # numerator: valid matched events at the truth level
    ismatched = datahandler.pass_reco & datahandler.pass_truth
    h_eff_numer = datahandler.compute_histogram(
        variable_truth, bins_truth, absoluteValue=absoluteValue, extra_cuts=ismatched
    )

    if isinstance(h_eff_denom, fh.FlattenedHistogram):
        histograms_obs_d['efficiency'] = h_eff_numer
        histograms_obs_d['efficiency'].divide(h_eff_denom)
    else:
        histograms_obs_d['efficiency'] = histUtils.divide(h_eff_numer, h_eff_denom)

    return histograms_obs_d

def compute_binned_efficiency_noflow(
    variable_reco,
    variable_truth,
    bins_reco,
    bins_truth,
    datahandler,
    datahandler_unmatched,
    absoluteValue = False,
    histograms_obs_d = {}
    ):
    if datahandler_unmatched is None:
        logger.warning("Cannot compute binned efficiency: unmatched data are not provided")
        return histograms_obs_d

    # denominator: all valid truth-level events, matched or unmatched
    if not 'generated' in histograms_obs_d:
        # compute truth-level distributions
        compute_truth_distributions(
            variable_truth, bins_truth,
            datahandler_sig = datahandler,
            datahandler_sig_unmatched = datahandler_unmatched,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

    h_eff_denom = histograms_obs_d['generated']

    # numerator: valid matched events at the truth level
    if isinstance(h_eff_denom, fh.FlattenedHistogram):
        h2d = datahandler.get_response_flattened(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValues = absoluteValue,
            normalize_truthbins = False,
            # TODO should use truth weights? 
            #weights = datahandler.get_weights(reco_level=False, valid_only=False)
        )

        h_eff_numer_noflow = h2d.projectToReco(flow=False)
        histograms_obs_d['efficiency_noflow'] = h_eff_numer_noflow
        histograms_obs_d['efficiency_noflow'].divide(h_eff_denom)
    else:
        h2d = datahandler.get_response(
            variable_reco, variable_truth,
            bins_reco, bins_truth,
            absoluteValue = absoluteValue,
            normalize_truthbins = False,
            # TODO should use truth weights? 
            #weights = datahandler.get_weights(reco_level=False, valid_only=False)
        )

        h_eff_numer_noflow = histUtils.projectToXaxis(h2d, flow=False)
        histograms_obs_d['efficiency_noflow'] = histUtils.divide(h_eff_numer_noflow, h_eff_denom)

    return histograms_obs_d

def compute_binned_corrections(
    variable_reco,
    variable_truth,
    bins_reco,
    bins_truth,
    datahandler,
    datahandler_unmatched,
    absoluteValue = False,
    histograms_obs_d = {},
    compute_acceptance = True,
    compute_efficiency = True,
    include_noflow = False
    ):

    # acceptance
    if compute_acceptance:
        logger.debug(f" binned acceptance corrections")
        compute_binned_acceptance(
            variable_reco, bins_reco,
            datahandler = datahandler,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

        if include_noflow:
            logger.debug(f" binned acceptance corrections (excluding overflow/underflow)")
            compute_binned_acceptance_noflow(
                variable_reco, variable_truth,
                bins_reco, bins_truth,
                datahandler = datahandler,
                absoluteValue = absoluteValue,
                histograms_obs_d = histograms_obs_d
            )

    # efficiency
    if compute_efficiency:
        logger.debug(f" binned efficiency corrections")
        compute_binned_efficiency(
            variable_truth, bins_truth,
            datahandler = datahandler,
            datahandler_unmatched =  datahandler_unmatched,
            absoluteValue = absoluteValue,
            histograms_obs_d = histograms_obs_d
        )

        if include_noflow:
            logger.debug(f" binned efficiency corrections (excluding overflow/underflow)")
            compute_binned_efficiency_noflow(
                variable_reco, variable_truth,
                bins_reco, bins_truth,
                datahandler = datahandler,
                datahandler_unmatched =  datahandler_unmatched,
                absoluteValue = absoluteValue,
                histograms_obs_d = histograms_obs_d
            )

def compute_unfolded_distributions(
    variables,
    bins,
    datahandler,
    unfolded_weights,
    absoluteValue = False,
    iteration = -1,
    histograms_obs_d = {},
    save_alliterations = False
    ):
    logger.debug(f" unfolded distributions")
    assert unfolded_weights.ndim == 2 # shape: (niterations, nevents)
    assert unfolded_weights.shape[0] > iteration

    weights_prior = datahandler.get_weights(valid_only=True, reco_level=False)
    assert len(weights_prior)==unfolded_weights.shape[-1]

    histograms_obs_d['unfolded'] = datahandler.compute_histogram(
        variables, bins,
        absoluteValue = absoluteValue,
        weights = unfolded_weights[iteration] * weights_prior
    )

    if save_alliterations:
        histograms_obs_d['unfolded_alliters'] = datahandler.compute_histogram(
            variables, bins,
            absoluteValue = absoluteValue,
            weights = unfolded_weights * weights_prior
        )

    return histograms_obs_d

def compute_unfolding_stat_uncertainty(
    variables,
    bins,
    datahandler,
    unfolded_weights_allruns,
    absoluteValue = False,
    histograms_obs_d = {},
    save_allruns = False
    ):
    logger.debug(f" unfolding stat correlation")
    assert unfolded_weights_allruns.ndim == 2 # shape: (nruns, nevents)

    weights_prior = datahandler.get_weights(valid_only=True, reco_level=False)
    assert len(weights_prior)==unfolded_weights_allruns.shape[-1]

    hists_unfolded_allruns = datahandler.compute_histogram(
        variables, bins,
        absoluteValue=absoluteValue,
        weights=unfolded_weights_allruns * weights_prior
    ) # return a list of histograms

    # store bin errors as histogram
    h_std_err = hists_unfolded_allruns[0].copy()

    if isinstance(h_std_err, fh.FlattenedHistogram):
        fhtmp = fh.average_histograms(hists_unfolded_allruns, standard_error_of_the_mean=True)
        std_errs = np.sqrt(fhtmp.flatten().variances())
        h_std_err.fromFlatArray(std_errs, None)

        # set the bin errors of the unfolded distribution
        if 'unfolded' in histograms_obs_d:
            h_unfolded_flat = histograms_obs_d['unfolded'].flatten()
            histUtils.set_hist_errors(h_unfolded_flat, std_errs)
            histograms_obs_d['unfolded'].fromFlat(h_unfolded_flat)

    else:
        htmp = histUtils.average_histograms(hists_unfolded_allruns, standard_error_of_the_mean=True)
        histUtils.set_hist_contents(h_std_err, np.sqrt(htmp.variances()))
        histUtils.set_hist_errors(h_std_err, 0)

        # set the bin errors of the unfolded distribution
        if 'unfolded' in histograms_obs_d:
            histUtils.set_hist_errors(histograms_obs_d['unfolded'], h_std_err.values())

    histograms_obs_d['network_unc'] = h_std_err

    if save_allruns:
        histograms_obs_d['unfolded_allruns'] = hists_unfolded_allruns

    # bin correlations
    if isinstance(h_std_err, fh.FlattenedHistogram):
        # flatten them first
        hists_unfolded_allruns_flat = [fh.flatten() for fh in hists_unfolded_allruns]
        histograms_obs_d['unfolded_correlation'] = histUtils.get_bin_correlations_from_hists(hists_unfolded_allruns_flat)
    else:
        histograms_obs_d['unfolded_correlation'] = histUtils.get_bin_correlations_from_hists(hists_unfolded_allruns)

    return histograms_obs_d

def compute_unfolded_distributions_ibu(
    histograms_obs_d,
    niterations,
    # for bootstrap data distribution
    datahandler_data,
    variables_reco,
    bins_reco,
    absoluteValue = False,
    nresamples = 25,
    #
    noflow = False,
    correct_acceptance = False,
    correct_efficiency = False,
    save_alliterations = False,
    ):
    logger.debug(f" Run IBU")

    # get histograms from histograms_obs_d
    # observed distribution
    if not 'reco_data' in histograms_obs_d:
        logger.error("Data reco-level distribution 'reco_data' not available. Aborting.")
        return histograms_obs_d
    h_obs = histograms_obs_d['reco_data']

    h_bkg = histograms_obs_d.get('reco_bkg')

    # response
    if not 'response' in histograms_obs_d:
        logger.error("Response 'response' not available. Aborting.")
        return histograms_obs_d
    resp = histograms_obs_d['response']

    # prior distribution
    if noflow:
        if not 'prior_noflow' in histograms_obs_d:
            logger.error("Prior distribution 'prior_noflow' not available. Aborting.")
            return histograms_obs_d
        h_prior = histograms_obs_d['prior_noflow']
    else:
        if not 'prior' in histograms_obs_d:
            logger.error("Prior distribution 'prior' not available. Aborting.")
            return histograms_obs_d
        h_prior = histograms_obs_d['prior']

    # bootstrap
    h_obs_resample = [
        datahandler_data.compute_histogram(
            variables_reco, bins_reco,
            absoluteValue=absoluteValue,
            bootstrap=True
        ) for _ in range(nresamples)
    ]

    # apply acceptance and/or efficiency corrections if needed
    acceptance = histograms_obs_d.get('acceptance') if correct_acceptance else None
    if correct_acceptance and acceptance is None:
        logger.warning("IBU: requested to apply acceptace corrections, but acceptance corrections are not available!")

    efficiency = histograms_obs_d.get('efficiency') if correct_efficiency else None
    if correct_efficiency and efficiency is None:
        logger.warning("IBU: requested to apply efficiency corrections, but efficiency corrections are not available")

    # run unfolding
    hists_ibu_alliters , bin_corr_ibu_alliters = ibu.run_ibu_from_histograms(
        resp, h_obs, h_prior, h_bkg,
        hist_obs_bootstrap = h_obs_resample,
        niterations = niterations,
        acceptance = acceptance,
        efficiency = efficiency,
        all_iterations = True
    )

    histograms_obs_d['ibu'] = hists_ibu_alliters[-1]
    histograms_obs_d['ibu_correlation'] = bin_corr_ibu_alliters[-1]

    if save_alliterations:
        histograms_obs_d['ibu_alliters'] = hists_ibu_alliters

    return histograms_obs_d

def compute_differential_xsections(
    histograms_obs_d,
    histname_unfolded,
    acceptance_corrected = True,
    efficiency_corrected = True,
    suffix = ""
    ):
    logger.debug(" Differential cross-sections")

    if not histname_unfolded in histograms_obs_d:
        logger.error(f"No unfolded distribution {histname_unfolded}!")
        return histograms_obs_d

    histname_corrected = f"{histname_unfolded}_corrected"

    if not acceptance_corrected:
        logger.warning("Acceptance correction has not been applied.")
        # nothing can be done at this stage

    if not efficiency_corrected:
        if 'efficiency' in histograms_obs_d:
            # apply efficiency corrections
            histograms_obs_d[histname_corrected] = apply_efficiency_correction(histograms_obs_d[histname_unfolded], histograms_obs_d['efficiency'])
        else:
            logger.warning("Efficiency correction is not available!")
            # still make a copy for now
            histograms_obs_d[histname_corrected] = histograms_obs_d[histname_unfolded].copy()
    else:
        histograms_obs_d[histname_corrected] = histograms_obs_d[histname_unfolded].copy()

    # compute differential cross-sections
    # absolute differential cross-sections
    histname_absdiff = f"absoluteDiffXs{suffix}"
    histograms_obs_d[histname_absdiff] = histograms_obs_d[histname_corrected].copy()
    histograms_obs_d[histname_absdiff] *= (1./ttbar_diffXs_run2_params['luminosity'])
    # divided by bin width
    if isinstance(histograms_obs_d[histname_absdiff], fh.FlattenedHistogram):
        histograms_obs_d[histname_absdiff].make_density()
    else:
        histograms_obs_d[histname_absdiff] = histograms_obs_d[histname_absdiff] / histUtils.get_hist_widths(histograms_obs_d[histname_absdiff])

    # normalized differential cross-sections
    histname_reldiff = f"relativeDiffXs{suffix}"
    histograms_obs_d[histname_reldiff] = histograms_obs_d[histname_corrected].copy()
    if isinstance(histograms_obs_d[histname_reldiff], fh.FlattenedHistogram):
        # normalize
        histograms_obs_d[histname_reldiff].renormalize(norm=1., density=False, flow=True)
        histograms_obs_d[histname_reldiff].make_density()
    else:
        histUtils.renormalize_hist(histograms_obs_d[histname_reldiff], norm=1.,  density=False, flow=True)
        histograms_obs_d[histname_reldiff] = histograms_obs_d[histname_reldiff] / histUtils.get_hist_widths(histograms_obs_d[histname_reldiff])

    return histograms_obs_d

def evaluate_metrics(
    observable,
    hists_dict,
    outdir = '.',
    plot = True
    ):
    mdict = {}
    mdict[observable] = {}

    logger.debug(f" Compute binned metrics")
    hists_uf_alliters = hists_dict[observable].get('unfolded_alliters')
    h_gen = hists_dict[observable].get('prior')
    h_truth = hists_dict[observable].get('truth')

    if hists_uf_alliters is None or h_gen is None:
        logger.error(f"Cannot compute binned metrics for {observable}")
    else:
        mdict[observable]["nominal"] = metrics.write_all_metrics_binned(
            hists_uf_alliters, h_gen, h_truth)

    hists_uf_all = hists_dict[observable].get('unfolded_all')
    if h_gen is not None and hists_uf_all is not None and len(hists_uf_all) > 1:
        # every run
        mdict[observable]['resample'] = metrics.write_all_metrics_binned(
            hists_uf_all, h_gen, h_truth)

    # IBU if available
    hists_ibu_alliters = hists_dict[observable].get('ibu_alliters')
    if hists_ibu_alliters is not None:
        mdict[observable]['IBU'] = metrics.write_all_metrics_binned(
            hists_ibu_alliters, h_gen, h_truth)

    # Save metrics to JSON file
    if outdir:
        metrics_dir = os.path.join(outdir, 'Metrics')
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)

        util.write_dict_to_json(mdict, metrics_dir + f"/{observable}.json")

        if plot:
            metrics.plot_all_metrics(
                mdict[observable], metrics_dir + f"/{observable}")

    return mdict

def apply_efficiency_correction(histogram, h_efficiency):
    if isinstance(histogram, list):
        return [ apply_efficiency_correction(hh, h_efficiency) for hh in histogram ]
    elif isinstance(histogram, fh.FlattenedHistogram):
        return histogram.divide(h_efficiency)
    else:
        # In case the correction histogram has a different binning
        # Get the correction factors using the histogram's bin centers
        f_eff = histUtils.read_histogram_at_locations(histogram.axes[0].centers, h_efficiency)
        return histogram * (1./f_eff)

def make_histograms_from_unfolder(
    unfolder,
    binning_config, # path to the binning config file
    observables, # list of observable names
    obsConfig_d, # dict for observable configurations
    iteration = -1, # by default take the last iteration
    nruns = None, # by default take all that are available
    outputdir = None, # str, output directory
    outfilename = "histograms.root", # str, output file name
    include_ibu = False, # If True, include also IBU for comparison
    compute_metrics = False, # If True, compute metrics
    plot_verbosity = 0, # int, control how many plots to make
    handle_unmatched = None
    ):

    # output directory
    if not outputdir:
        outputdir = unfolder.outdir

    # in case it is not the last iteration that is used
    if iteration != -1:
        outputdir = os.path.join(outputdir, f"iter{iteration+1}")
        # +1 because the index 0 of the weights array is for iteration 1

    # in case not all runs are used to make histograms
    if nruns is not None:
        outputdir = os.path.join(outputdir, f"nruns{nruns}")

    if not os.path.isdir(outputdir):
        logger.info(f"Create directory {outputdir}")
        os.makedirs(outputdir)

    # control flags
    #all_runs = nruns is None # if number of runs is explicitly specified, no need to include all runs
    include_reco = True
    all_iterations = compute_metrics or plot_verbosity >= 2
    all_histograms = compute_metrics or plot_verbosity >= 2

    ######
    # get data handlers from the unfolder
    dh_sig = unfolder.handle_sig
    dh_obs = unfolder.handle_obs
    dh_bkg = unfolder.handle_bkg
    dh_obsbkg = unfolder.handle_obsbkg

    # unmatched signal events for compute binned efficiency correction if needed
    dh_sig_um = handle_unmatched

    ######
    # unfolded weights
    unfolded_weights_all = unfolder.get_unfolded_weights(None, nruns) # shape: (nruns, niterations, nevents)
    unfolded_weights_nominal = np.median(unfolded_weights_all, axis=0) # shape: (niterations, nevents)

    reco_weights_all = unfolder.get_reco_weights(None, nruns) # shape: (nruns, niterations, nevents)
    reco_weights_nominal = np.median(reco_weights_all, axis=0) if reco_weights_all is not None else None
    # shape: (niterations, nevents)

    ######
    # make histograms
    # binning
    binCfg_d = util.get_bins_dict(binning_config)

    # a dictionary to store histograms
    histograms_d = {}

    for obs in observables:
        logger.info(f"Making histograms: {obs}")
        histograms_d[obs] = {}

        # variables
        obs_list = obs.split("_vs_")
        vnames_reco = [obsConfig_d[ob]['branch_det'] for ob in obs_list]
        vnames_truth = [obsConfig_d[ob]['branch_mc'] for ob in obs_list]

        # binning
        # TODO different binning at reco and truth level
        bins_reco = binCfg_d[obs]
        bins_truth = binCfg_d[obs]

        # absolute values
        isAbsolute = ["_abs" in ob for ob in obs_list]

        ###
        # reco level
        if include_reco:
            compute_reco_distributions(
                vnames_reco, bins_reco,
                datahandler_sig = dh_sig,
                datahandler_data = dh_obs,
                datahandler_bkg = dh_bkg,
                datahandler_databkg = dh_obsbkg,
                absoluteValue = isAbsolute,
                histograms_obs_d = histograms_d[obs]
            )

            if all_iterations and reco_weights_nominal is not None:
                compute_reco_distributions_alliters(
                    vnames_reco, bins_reco,
                    reco_weights_alliters = reco_weights_nominal,
                    datahandler_sig = dh_sig,
                    absoluteValue = isAbsolute,
                    histograms_obs_d = histograms_d[obs]
                )

        ###
        # truth level
        truth_known = dh_obs is not None and dh_obs.data_truth is not None
        compute_truth_distributions(
            vnames_truth, bins_truth,
            datahandler_sig = dh_sig,
            datahandler_data = dh_obs if truth_known else None,
            datahandler_sig_unmatched = dh_sig_um,
            absoluteValue = isAbsolute,
            histograms_obs_d = histograms_d[obs]
        )

        # exclude events in overflow and underflow bins
        compute_truth_distributions_noflow(
            vnames_reco, vnames_truth,
            bins_reco, bins_truth,
            datahandler_sig = dh_sig,
            datahandler_data = dh_obs if truth_known else None,
            absoluteValue = isAbsolute,
            histograms_obs_d = histograms_d[obs]
        )

        ###
        # response
        compute_response(
            vnames_reco, vnames_truth,
            bins_reco, bins_truth,
            datahandler = dh_sig,
            absoluteValue = isAbsolute,
            histograms_obs_d = histograms_d[obs]
        )

        ###
        # binned acceptance and efficiency corrections
        compute_binned_corrections(
            vnames_reco, vnames_truth,
            bins_reco, bins_truth,
            datahandler = dh_sig,
            datahandler_unmatched = dh_sig_um,
            absoluteValue = isAbsolute,
            histograms_obs_d = histograms_d[obs],
            compute_acceptance = unfolder.with_acceptance_correction,
            compute_efficiency = dh_sig_um is not None,
            include_noflow = True
        )

        ###
        # unfolded distributions
        compute_unfolded_distributions(
            vnames_truth, bins_truth,
            datahandler = dh_sig,
            unfolded_weights = unfolded_weights_nominal,
            absoluteValue = isAbsolute,
            iteration = iteration,
            histograms_obs_d = histograms_d[obs],
            save_alliterations = all_iterations
        )

        compute_unfolding_stat_uncertainty(
            vnames_truth, bins_truth,
            datahandler = dh_sig,
            unfolded_weights_allruns = unfolded_weights_all[:,iteration,:],
            absoluteValue = isAbsolute,
            histograms_obs_d = histograms_d[obs],
            save_allruns = all_histograms
        )

        # differential cross-sections
        # only compute diff. Xs if acceptance corrections have been applied 
        # and efficiency corrections either have been applied or are available
        # and it is possible to compute efficiency corrections if they have not been applied yet
        compute_diffXs = unfolder.with_acceptance_correction and (unfolder.with_efficiency_correction or "efficiency" in histograms_d[obs])
        if compute_diffXs:
            compute_differential_xsections(
                histograms_obs_d = histograms_d[obs],
                histname_unfolded = 'unfolded',
                acceptance_corrected = unfolder.with_acceptance_correction,
                efficiency_corrected = unfolder.with_efficiency_correction
            )

        ###
        # ibu
        if include_ibu:
            niterations = iteration + 1
            if iteration < 0:
                niterations += unfolded_weights_nominal.shape[0]

            compute_unfolded_distributions_ibu(
                histograms_obs_d = histograms_d[obs],
                niterations = niterations,
                datahandler_data = dh_obs,
                variables_reco = vnames_reco,
                bins_reco = bins_reco,
                absoluteValue = isAbsolute,
                noflow = False,
                correct_acceptance = unfolder.with_acceptance_correction,# apply same corrections as OmniFold
                correct_efficiency = unfolder.with_efficiency_correction,# apply same corrections as OmniFold
                save_alliterations = all_iterations,
            )

            if unfolder.with_acceptance_correction and (unfolder.with_efficiency_correction or dh_sig_um is not None):
                # only compute diff. Xs if acceptance corrections have been applied 
                # and it is possible to compute efficiency corrections if they have not been applied
                compute_differential_xsections(
                    histograms_obs_d = histograms_d[obs],
                    histname_unfolded = 'ibu',
                    acceptance_corrected = unfolder.with_acceptance_correction,
                    efficiency_corrected = unfolder.with_efficiency_correction,
                    suffix = '_ibu'
                )

        # compute metrics
        if compute_metrics and len(obs_list)==1:
            evaluate_metrics(
                obs,
                histograms_d,
                outdir = outputdir,
                plot = plot_verbosity > 1
            )

    # save histograms to file
    outname_hist = os.path.join(outputdir, outfilename)
    logger.info(f"Write histograms to file: {outname_hist}")
    histUtils.write_histograms_dict_to_file(histograms_d, outname_hist)

    # plot
    outdir_plot = os.path.join(outputdir, 'plots')
    if not os.path.isdir(outdir_plot):
        os.makedirs(outdir_plot)

    plot_histograms_from_dict(
        histograms_d,
        outputdir = outdir_plot,
        obsConfig_d = obsConfig_d,
        include_ibu = include_ibu,
        plot_verbosity = plot_verbosity
    )

def load_unmatched_datahandler(
    filepaths_unmatched,
    vnames_truth,
    outputdir = '.',
    weight_type = 'nominal',
    filepaths_signal=None,
    suffix=''
    ):

    if not filepaths_unmatched:
        if filepaths_signal is None:
            logger.error("File paths to unmatched signal samples unknown!")
        else:
            # guess file paths of unmatched signal samples based on the file paths of the signal samples
            # 'some_name.h5' --> 'some_name_unmatched_truth.h5'
            filepaths_unmatched = ["{}_unmatched_truth{}".format(*os.path.splitext(fnm)) for fnm in filepaths_signal]

    try:
        dh_umnatched = DataHandlerH5(
            filepaths_unmatched,
            variable_names_mc = vnames_truth,
            outputname = os.path.join(outputdir, f'sig_unmatched{suffix}'),
            weight_type = weight_type,
            use_existing_vds = False,
        )
    except Exception as ex:
        dh_umnatched = None
        logger.warning(f"Failed to load unmatched signal samples from {filepaths_unmatched}: {ex}")

    return dh_umnatched

def make_histograms(
    filepath_args,
    binning_config,
    observables = [],
    observable_config = '',
    iterations = [-1],
    nruns = [None],
    outputdir = None,
    outfilename = 'histograms.root',
    include_ibu = False,
    compute_metrics = False,
    plot_verbosity = 0,
    verbose = False,
    apply_corrections = False,
    filepaths_unmatched = []
    ):
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    tracemalloc.start()

    ####
    # open the argument config file
    if os.path.isdir(filepath_args): # for backward compatibility
        filepath_args = os.path.join(filepath_args, "arguments.json")
    try:
        args_d = util.read_dict_from_json(filepath_args)
    except:
        logger.critical(f"Failed to read argument config {filepath_args}")

    if outputdir is None:
        outputdir = os.path.dirname(filepath_args)

    # update arguments if needed
    if observable_config:
        args_d['observable_config'] = observable_config

    if observables:
        obs_1d_unique = set()
        for obs in observables:
            # pt, pt_vs_phi, etc.
            for v in obs.split("_vs_"):
                obs_1d_unique.add(v)

        args_d['observables'] = list(obs_1d_unique)
    else:
        observables = list(set(args_d['observables']+args_d['observables_extra']))

    # read observable configurations
    obsCfg_d = util.read_dict_from_json(args_d['observable_config'])

    # load data handlers via the unfolder
    logger.info(f"Load data handlers")
    t_load_start = time.time()

    args_d['suffix'] = "_hist"
    ufdr = load_unfolder(args_d)

    # unmatched signal events for binned efficiency corrections if needed
    dh_sig_um = None
    if apply_corrections:
        dh_sig_um = load_unmatched_datahandler(
            filepaths_unmatched = filepaths_unmatched,
            vnames_truth = [obsCfg_d[obs]['branch_mc'] for obs in args_d['observables']],
            outputdir = outputdir,
            weight_type = args_d.get("weight_mc", 'nominal'),
            filepaths_signal = args_d['signal'],
            suffix = "_hist"
        )

        if dh_sig_um is None:
            logger.error(f"Failed to load unmatched signal events. Cannot compute binned efficiency corrections!")

    t_load_stop = time.time()
    logger.info(f"Done")
    logger.debug(f"Loading time: {(t_load_stop-t_load_start):.2f} seconds")
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    logger.info("Start histogramming")
    t_hist_start = time.time()

    # check if iterations is a list
    if not isinstance(iterations, list):
        iterations = [iterations]

    # check if nruns is a list
    if not isinstance(nruns, list):
        nruns = [nruns]

    for it in iterations:
        logger.info(f"iteration: {it}")

        for n in nruns:
            logger.info(f" nruns: {n}")

            make_histograms_from_unfolder(
                ufdr,
                binning_config,
                observables,
                obsCfg_d,
                iteration = it,
                nruns = n,
                outputdir = outputdir,
                outfilename = outfilename,
                include_ibu = include_ibu,
                compute_metrics = compute_metrics,
                plot_verbosity = plot_verbosity,
                handle_unmatched = dh_sig_um,
            )

    t_hist_stop = time.time()
    logger.debug(f"Histogramming time: {(t_hist_stop-t_hist_start):.2f} seconds")
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*1e-6:.1f} MB; Peak usage: {mpeak*1e-6:.1f} MB")

    tracemalloc.stop()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Make histograms for unfolding results")

    parser.add_argument('filepath_args', type=str, action=util.ParseEnvVar,
                        help="Path to the configuration file (generally the same as the one used for unfolding).")
    parser.add_argument("--binning-config", type=str, action=util.ParseEnvVar,
                        default='${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json',
                        help="Path to the binning config file for variables.")
    parser.add_argument("--observables", nargs='+', default=[],
                        help="List of observables to make histograms. If provided, overwrite the ones from filepath_args.")
    parser.add_argument("--observable-config", type=str, action=util.ParseEnvVar,
                        help="Path to the observable config file. If provided, overwrite the ones from filepath_args.")
    parser.add_argument("-i", "--iterations", type=int, nargs='+', default=[-1],
                        help="Use the results at the specified iteration.")
    parser.add_argument("-n", "--nruns", type=int, nargs='+', default=[None],
                        help="Number of runs for making unfolded distributions. If None, use all that are available.")
    parser.add_argument("-o", "--outputdir", type=str, action=util.ParseEnvVar,
                        help="Output directory. If not provided, use the directory of filepath_args.")
    parser.add_argument("-f", "--outfilename", type=str, default="histograms.root",
                        help="Output file name.")
    parser.add_argument('--include-ibu', action='store_true',
                        help="If True, run unfolding also with IBU.")
    parser.add_argument('--compute-metrics', action='store_true',
                        help="If True, compute metrics of unfolding performance.")
    parser.add_argument('-p', '--plot-verbosity', action='count', default=0,
                        help="Plot verbose level. '-ppp' to make all plots.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="If True, set logging level to DEBUG, otherwise INFO.")
    parser.add_argument("-u", "--filepaths-unmatched", 
                        type=str, nargs='+', action=util.ParseEnvVar, default=[],
                        help="File paths to the signal MC samples that are not truth matched. If not provided, infer from signal MC samples names.")
    parser.add_argument("-c", "--apply-corrections", action='store_true',
                        help="If True, apply binned efficiency corrections.")

    args = parser.parse_args()

    util.configRootLogger()

    make_histograms(**vars(args))