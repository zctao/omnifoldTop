import os
import logging

import util
import histogramming as myhu
import FlattenedHistogram as fh
from ttbarDiffXsRun2.plotDiffXs import plot_diffXs_1D, plot_diffXs_2D, plot_diffXs_3D, rescale_oom

util.configRootLogger()
logger = logging.getLogger("compare")

obsNameMap = {
    "mtt" : "ttbar_m",
    "ptt" : "ttbar_pt",
    "ytt" : "ttbar_y",
    "ytt_abs" : "ttbar_abs_y",
    "th_pt" : "top_had_pt",
    "tl_pt" : "top_lep_pt",
    "th_y" : "top_had_y",
    "th_y_abs" : "top_had_abs_y",
    "tl_y" : "top_lep_y",
    "tl_y_abs" : "top_lep_abs_y",
    "ptt_vs_mtt" : "ttbar_pt_ttbar_m",
    "th_pt_vs_mtt" : "top_had_pt_ttbar_m",
    "ytt_abs_vs_mtt" : "ttbar_abs_y_ttbar_m",
    "ptt_vs_ytt_abs" : "ttbar_pt_ttbar_abs_y",
    "mtt_vs_ytt_abs" : "ttbar_m_ttbar_abs_y",
    "mtt_vs_ptt_vs_ytt_abs" : "ttbar_m_ttbar_pt_ttbar_absy",
    "mtt_vs_th_pt_vs_th_y_abs" : "ttbar_m_top_had_pt_top_had_absy",
    "mtt_vs_th_pt_vs_ytt_abs" : "ttbar_m_top_had_pt_ttbar_absy",
    "mtt_vs_th_y_abs_vs_ytt_abs" : "ttbar_m_top_had_absy_ttbar_absy"
}

histNameMap = {
    "absoluteDiffXs" : "AbsoluteDiffXs",
    "relativeDiffXs" : "RelativeDiffXs",
    "unfolded_corrected" : "DataUnfolded",
    "acceptance" : "Acceptance",
    "efficiency" : "Efficiency",
    "reco_data" : "Data",
    "reco_bkg" : "Background",
    "reco_sig" : "SignalReco",
    "prior" : "SignalTruth",
    "response" : "Response"
}

def compare_histograms(
    figname,
    observable_labels,
    histogram_omnifold,
    histogram_binned,
    binerors_omnifold_d = {}, # relative bin errors
    #binerrors_binned = None,
    label_omnifold = "MultiFold",
    label_binned = "RooUnfold IBU",
    ylabel = '',
    yscale_log = False,
    rescales_order_of_magnitude = None
    ):

    logger.debug(f" Create plot: {figname}")

    ndim = len(observable_labels)

    if ndim == 1:
        plot_diffXs_1D(
            figname,
            observable_labels,
            histogram_omnifold,
            [histogram_binned],
            label_omnifold,
            [label_binned],
            hists_relerrs_dict = binerors_omnifold_d,
            ylabel = ylabel,
            ylabel_ratio = f"Ratio to {label_omnifold}",
            log_obs = False,
            log_diffXs = yscale_log,
        )
    else:
        assert isinstance(histogram_omnifold, fh.FlattenedHistogram)

        # convert histogram_binned to fh.FlattenedHistogram
        # assume binnings are consistent
        fhist_binned = histogram_omnifold.copy()
        fhist_binned.fromFlatArray(histogram_binned.values(), histogram_binned.variances())

        if ndim == 2:
            f_plot = plot_diffXs_2D
        elif ndim == 3:
            f_plot = plot_diffXs_3D
        else:
            raise RuntimeError(f"Cannot handle {ndim} dimensional observables {observable_labels}")

        f_plot(
            figname,
            observable_labels,
            histogram_omnifold,
            [fhist_binned],
            label_omnifold,
            [label_binned],
            hists_relerrs_dict = binerors_omnifold_d,
            ylabel = ylabel,
            ylabel_ratio = f"Ratio to {label_omnifold}",
            log_obs = False,
            log_diffXs = yscale_log,
            rescales_order_of_magnitude = rescales_order_of_magnitude
        )

def get_histogram_binned(
    observables,
    topdir, # str, e.g. "/mnt/xrootdg/ztao/fromDavide/MINI_362_eos/",
    prefix = "truth_4j2b_ljets_PseudoTop_Reco"
    ):

    obs_list = observables.split("_vs_")
    ndim = len(obs_list)

    if ndim == 1:
        hists_dir = os.path.join(topdir, "top_observable_MINI_unfolding", "truth", "4j2b_ljets")
        suffix = "central_ljets.root"
    elif ndim == 2:
        hists_dir = os.path.join(topdir, "top_observable_MINI_unfolding_2D", "truth", "4j2b_ljets")
        suffix = "multi_central_ljets.root"
    elif ndim == 3:
        hists_dir = os.path.join(topdir, "top_observable_MINI_unfolding_3D", "truth", "4j2b_ljets")
        suffix = "multi_3D_central_ljets.root"
    else:
        raise RuntimeError(f"Cannot handle observable {observables}")

    obsname = obsNameMap.get(observables)
    if not obsname:
        raise RuntimeError(f"Cannot convert observable {observables}")

    fpath_hist = os.path.join(hists_dir, f"{prefix}_{obsname}_{suffix}")

    # check if the file exists
    if not os.path.isfile(fpath_hist):
        raise RuntimeError(f"Cannot find histogram file {fpath_hist}")
    
    return myhu.read_histograms_dict_from_file(fpath_hist)

def compare_wrt_binned(
    fpath_omnifold,
    binned_results_topdir = "/mnt/xrootdg/ztao/fromDavide/MINI_362/",
    binned_histogram_prefix = "truth_4j2b_ljets_PseudoTop_Reco",
    fpath_binerrors_absolute = None,
    fpath_binerrors_relative = None,
    observables = [],
    output_dir = './comparison',
    observable_config = "configs/observables/vars_ttbardiffXs_pseudotop.json"
    ):

    if not os.path.isdir(output_dir):
        logger.info(f"Create output directory: {output_dir}")
        os.makedirs(output_dir)

    logger.debug(f"Read observable config from {observable_config}")
    obsConfig_d = util.read_dict_from_json(observable_config)

    ######
    # OmniFold results
    logger.info(f"Retrieve OmniFold histograms from {fpath_omnifold}")
    hists_omnifold_d = myhu.read_histograms_dict_from_file(fpath_omnifold)

    # uncertainties
    # absolute
    if fpath_binerrors_absolute:
        logger.info(f"Retrieve OmniFold bin errors for absolute diff. Xs from {fpath_binerrors_absolute}")
        binerrors_abs_omnifold_d = myhu.read_histograms_dict_from_file(fpath_binerrors_absolute)
    else:
        binerrors_abs_omnifold_d = None

    # relative/normalized
    if fpath_binerrors_relative:
        logger.info(f"Retrieve OmniFold bin errors for relative diff. Xs from {fpath_binerrors_relative}")
        binerrors_rel_omnifold_d = myhu.read_histograms_dict_from_file(fpath_binerrors_relative)
    else:
        binerrors_rel_omnifold_d = None

    # loop over observables
    if not observables:
        observables = hists_omnifold_d.keys()

    for obs in observables:
        logger.info(f"{obs}")

        outdir_obs = os.path.join(output_dir, obs)
        if not os.path.isdir(outdir_obs):
            logger.debug(f"Create directory: {outdir_obs}")

        # Binned method results for different observables are stored in different files
        logger.info(f" Retrieve binned results")
        try:
            hists_binned_obs_d = get_histogram_binned(
                obs,
                topdir = binned_results_topdir,
                prefix = binned_histogram_prefix
            )
        except RuntimeError as ex:
            logger.error(f" Failed to get the binned results: {ex}")
            continue

        # Plot comparison
        obs_list = obs.split('_vs_')
        obs_labels = util.get_obs_label(obs_list, obsConfig_d)

        if len(obs_list) > 1:
            yscale_log = True
        else:
            yscale_log = obsConfig_d[obs_list[0]].get("log_scale", False)

        rescale_oom_obs = rescale_oom.get(obs)

        # inputs
        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_signal"),
                obs_labels,
                hists_omnifold_d[obs]['reco_sig'],
                hists_binned_obs_d.get(histNameMap['reco_sig']),
                ylabel = "Events",
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_data"),
                obs_labels,
                hists_omnifold_d[obs]['reco_data'],
                hists_binned_obs_d.get(histNameMap['reco_data']),
                ylabel = "Events",
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_background"),
                obs_labels,
                hists_omnifold_d[obs]['reco_bkg'],
                hists_binned_obs_d.get(histNameMap['reco_bkg']),
                ylabel = "Events",
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_unfolded"),
                obs_labels,
                hists_omnifold_d[obs]['unfolded_corrected'],
                hists_binned_obs_d.get(histNameMap['unfolded_corrected']),
                ylabel = "Events",
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_efficiency"),
                obs_labels,
                hists_omnifold_d[obs]['efficiency'],
                hists_binned_obs_d.get(histNameMap['efficiency']),
                ylabel = "",
                yscale_log = False,
                rescales_order_of_magnitude = None
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_prior"),
                obs_labels,
                hists_omnifold_d[obs]['prior'],
                hists_binned_obs_d.get(histNameMap['prior']),
                ylabel = "Events",
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        # differential cross-sections
        # absolute
        relerrs_abs_omf_d = {}
        if binerrors_abs_omnifold_d:
            # total uncertainty
            relerrs_abs_omf_d['Total'] = (
                binerrors_abs_omnifold_d[obs]['Total'].get('total_up'),
                binerrors_abs_omnifold_d[obs]['Total'].get('total_down')
            )

            # unfolding uncertainty
            relerrs_abs_omf_d['Unfold'] = (
                binerrors_abs_omnifold_d[obs]['Total'].get('Unfold_up'),
                binerrors_abs_omnifold_d[obs]['Total'].get('Unfold_down')
            )

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_absDiffXs"),
                obs_labels,
                hists_omnifold_d[obs]['absoluteDiffXs'],
                hists_binned_obs_d.get(histNameMap['absoluteDiffXs']),
                binerors_omnifold_d = relerrs_abs_omf_d,
                ylabel = util.get_diffXs_label(obs_list, obsConfig_d, False, "pb"),
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

        # relative
        relerrs_rel_omf_d = {}
        if binerrors_abs_omnifold_d:
            # total uncertainty
            relerrs_rel_omf_d['Total'] = (
                binerrors_rel_omnifold_d[obs]['Total'].get('total_up'),
                binerrors_rel_omnifold_d[obs]['Total'].get('total_down')
            )

            # unfolding uncertainty
            relerrs_rel_omf_d['Unfold'] = (
                binerrors_rel_omnifold_d[obs]['Total'].get('Unfold_up'),
                binerrors_rel_omnifold_d[obs]['Total'].get('Unfold_down')
            )

        try:
            compare_histograms(
                os.path.join(outdir_obs, "compare_relDiffXs"),
                obs_labels,
                hists_omnifold_d[obs]['relativeDiffXs'],
                hists_binned_obs_d.get(histNameMap['relativeDiffXs']),
                binerors_omnifold_d = relerrs_rel_omf_d,
                ylabel = util.get_diffXs_label(obs_list, obsConfig_d, True, "pb"),
                yscale_log = yscale_log,
                rescales_order_of_magnitude = rescale_oom_obs
            )
        except Exception as ex:
            logger.error(f"Failed to compare 'reco_sig': {ex}")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpath_omnifold", type=str,
                        help="Path to histograms made from OmniFold weights")
    parser.add_argument("--binned-results-topdir", type=str,
                        default="/mnt/xrootdg/ztao/fromDavide/MINI_362_eos/",
                        help="Path to directory of binned results")
    parser.add_argument("--binned-histogram-prefix", type=str, 
                        default="truth_4j2b_ljets_PseudoTop_Reco",
                        help="Prefix of the histograms from binned method")
    parser.add_argument("--fpath-binerrors-absolute", type=str,
                        help="Path to the relative bin errors of the absolute differential cross-sections from OmniFold")
    parser.add_argument("--fpath-binerrors-relative", type=str,
                        help="Path to the relative bin errors of the relative differential cross-sections from OmniFold")
    parser.add_argument("--observables", nargs="+", type=str, default=[],
                        help="List of observables to compare. If not specified, take all that are available")
    parser.add_argument("-o", "--output-dir", type=str, default="./comparison",
                        help="Output directory")
    parser.add_argument(
        "--observable-config", type=str, action=util.ParseEnvVar,
        default="${SOURCE_DIR}/configs/observables/vars_ttbardiffXs_pseudotop.json",
        help="Path to the observable config file")

    args = parser.parse_args()

    compare_wrt_binned(**vars(args))