import os
import numpy as np
import uproot

import plotter
import histogramming as myhu
import util
from ttbarDiffXsRun2.helpers import fileNameMap

# plot style
draw_opt_dm = {'label':'DM', 'color':'black', 'xerr':True, 'histtype':'step', 'linewidth':1}
draw_opt_zt = {'label':'ZT', 'color':'tab:red', 'xerr':True, 'histtype':'step', 'linewidth':1}

def get_histogram_dm(observable, histname, results_dir):
    fpath = fileNameMap.get(observable)
    if fpath is None:
        print(f"ERROR: no file for {observable}")
        return

    fpath = os.path.join(results_dir, fpath)

    h = None
    with uproot.open(fpath) as f:
        h = f[histname].to_hist()

    return h

def get_histogram_of(observable, histname, results_dir, iteration=None):

    fpath = results_dir
    if iteration:
        fpath = os.path.join(fpath, f"iter{iteration}")

    fpath = os.path.join(fpath, "histograms.root")

    assert(os.path.isfile(fpath))

    h = None
    with uproot.open(fpath) as f:
        h = f[f"{observable}/{histname}"].to_hist()

    return h

def compare_inputs(
    observable,
    indir_dm,
    indir_zt,
    outdir,
    obsConfig_d,
    iteration=None,
    normalize=False
    ):

    # Data
    h_data_dm = get_histogram_dm(observable, "Data", indir_dm)
    h_data_zt = get_histogram_of(observable, "reco_data", indir_zt, iteration)

    if normalize:
        myhu.renormalize_hist(h_data_dm, density=False)
        myhu.renormalize_hist(h_data_zt, density=False)

    figname_obs = os.path.join(outdir, f"{observable}_data")
    if normalize:
        figname_obs = figname_obs+"_normalized"

    print(f"Create plot: {figname_obs}")
    plotter.plot_histograms_and_ratios(
        figname_obs,
        hists_numerator = [h_data_zt],
        draw_options_numerator = [draw_opt_zt],
        hist_denominator = h_data_dm,
        draw_option_denominator = draw_opt_dm,
        xlabel = obsConfig_d[observable]['xlabel'],
        ylabel = "Events" if not normalize else "Normalized Events",
        ylabel_ratio = "Ratio to DM",
        title = "Data"
        )

    # Background
    h_bkg_dm = get_histogram_dm(observable, "Background", indir_dm)
    h_bkg_zt = get_histogram_of(observable, "reco_bkg", indir_zt, iteration)

    if normalize:
        myhu.renormalize_hist(h_bkg_dm, density=False)
        myhu.renormalize_hist(h_bkg_zt, density=False)

    figname_bkg = os.path.join(outdir, f"{observable}_bkg")
    if normalize:
        figname_bkg = figname_bkg+"_normalized"

    print(f"Create plot: {figname_bkg}")
    plotter.plot_histograms_and_ratios(
        figname_bkg,
        hists_numerator = [h_bkg_zt],
        draw_options_numerator = [draw_opt_zt],
        hist_denominator = h_bkg_dm,
        draw_option_denominator = draw_opt_dm,
        xlabel = obsConfig_d[observable]['xlabel'],
        ylabel = "Events" if not normalize else "Normalized Events",
        ylabel_ratio = "Ratio to DM",
        title = "Background"
        )

    # Signal
    h_sig_dm = get_histogram_dm(observable, "SignalReco", indir_dm)
    h_sig_zt = get_histogram_of(observable, "reco_sig", indir_zt, iteration)

    if normalize:
        myhu.renormalize_hist(h_sig_dm, density=False)
        myhu.renormalize_hist(h_sig_zt, density=False)

    figname_sig = os.path.join(outdir, f"{observable}_sig")
    if normalize:
        figname_sig = figname_sig+"_normalized"

    print(f"Create plot: {figname_sig}")
    plotter.plot_histograms_and_ratios(
        figname_sig,
        hists_numerator = [h_sig_zt],
        draw_options_numerator = [draw_opt_zt],
        hist_denominator = h_sig_dm,
        draw_option_denominator = draw_opt_dm,
        xlabel = obsConfig_d[observable]['xlabel'],
        ylabel = "Events" if not normalize else "Normalized Events",
        ylabel_ratio = "Ratio to DM",
        title = "Signal Reco"
        )

    # Signal truth
    h_sig_mc_dm = get_histogram_dm(observable, "SignalTruth", indir_dm)
    h_sig_mc_zt = get_histogram_of(observable, "prior", indir_zt, iteration)

    if normalize:
        myhu.renormalize_hist(h_sig_mc_dm, density=False)
        myhu.renormalize_hist(h_sig_mc_zt, density=False)

    figname_prior = os.path.join(outdir, f"{observable}_prior")
    if normalize:
        figname_prior = figname_prior+"_normalized"

    print(f"Create plot: {figname_prior}")
    plotter.plot_histograms_and_ratios(
        figname_prior,
        hists_numerator = [h_sig_mc_zt],
        draw_options_numerator = [draw_opt_zt],
        hist_denominator = h_sig_mc_dm,
        draw_option_denominator = draw_opt_dm,
        xlabel = obsConfig_d[observable]['xlabel'],
        ylabel = "Events" if not normalize else "Normalized Events",
        ylabel_ratio = "Ratio to DM",
        title = "Signal Truth"
        )

def compare_response(observable, indir_dm, indir_zt, outdir):
    # Response
    # DM
    response_dm = get_histogram_dm(observable, "Response", indir_dm)
    response_dm_norm = response_dm.copy()
    response_dm_norm.view()['value'] = response_dm.values() / response_dm.values().sum(axis=0)

    figname_dm = os.path.join(outdir, f"{observable}_resp_dm")
    print(f"Create plot: {figname_dm}")
    plotter.plot_response(figname_dm, response_dm_norm, observable)

    # ZT
    response_zt = get_histogram_of(observable, "response", indir_zt)
    figname_zt = os.path.join(outdir, f"{observable}_resp_zt")
    print(f"Create plot: {figname_zt}")
    plotter.plot_response(figname_zt, response_zt, observable)

    # Ratio
    response_ratio = response_zt.copy()
    response_ratio.view()['value'] = response_zt.values() / response_dm_norm.values()
    figname_ratio = os.path.join(outdir, f"{observable}_resp_ratio")
    plotter.plot_response(figname_ratio, response_ratio, observable, title="Ratio")

def compare_diffXs(
    observable,
    indir_dm,
    indir_zt,
    outdir,
    obsConfig_d,
    iteration=None
    ):

    # Relative diff. Xs
    # DM
    relDiffXs_dm = get_histogram_dm(observable, "RelativeDiffXs", indir_dm)

    # ZT
    relDiffXs_zt = get_histogram_of(observable, "relativeDiffXs", indir_zt, iteration)

    # ZT IBU
    relDiffx_ibu_zt = get_histogram_of(observable, "relativeDiffXs_ibu", indir_zt, iteration)

    # plot
    draw_opt_zt_ibu = draw_opt_zt.copy()
    draw_opt_zt_ibu.update({'label':'ZT (IBU)', 'color':'tab:blue'})

    figname_diffxs = os.path.join(outdir, f"{observable}_reldiffxs")
    print(f"Create plot: {figname_diffxs}")
    plotter.plot_histograms_and_ratios(
        figname_diffxs,
        hists_numerator = [relDiffx_ibu_zt, relDiffXs_zt],
        draw_options_numerator = [draw_opt_zt_ibu, draw_opt_zt],
        hist_denominator = relDiffXs_dm,
        draw_option_denominator = draw_opt_dm,
        xlabel = obsConfig_d[observable]['xlabel'],
        ylabel = "",
        ylabel_ratio = "Ratio to DM",
        title = "Relative Diff. XS",
        log_scale = True,
        #ratio_lim = (0.9,1.1)
        )

def compare_run2_DM(
    observables,
    results_dir,
    results_ref,
    outdir,
    observable_config,
    iteration
    ):

    if not os.path.isdir(outdir):
        print(f"Create directory {outdir}")
        os.makedirs(outdir)

    obsCfg_d = util.read_dict_from_json(observable_config)

    for ob in observables:
        print(f"{ob}")

        # inputs
        compare_inputs(
            ob,
            indir_dm = results_ref,
            indir_zt = results_dir,
            outdir = outdir,
            obsConfig_d = obsCfg_d,
            iteration = iteration,
            normalize=False
            )

        # response
        compare_response(
            ob,
            indir_dm = results_ref,
            indir_zt = results_dir,
            outdir = outdir
            )

        # relative differential cross section
        compare_diffXs(
            ob,
            indir_dm = results_ref,
            indir_zt = results_dir,
            outdir = outdir,
            obsConfig_d = obsCfg_d,
            iteration = iteration
            )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--observables", nargs="+", type=str,
                        default=['th_pt', 'th_y', 'mtt'])
    parser.add_argument("--results-dir", type=str,
                        default="/home/ztao/data/OmniFoldOutputs/Run2/nominal/output_run2_ljets")
    parser.add_argument("--results-ref", type=str,
                        default="/mnt/xrootdg/ztao/fromDavide/4j2b_ljets_central")
    parser.add_argument("--observable-config", type=str,
                        default="configs/observables/vars_ttbardiffXs.json")
    parser.add_argument("-i", "--iteration", type=int)
    parser.add_argument("-o", "--outdir", type=str, default="compare_diffxs")

    args = parser.parse_args()

    compare_run2_DM(**vars(args))
