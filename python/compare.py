#!/usr/bin/env python3
import os
import numpy as np

from datahandler import DataHandler
from omnifoldwbkg import OmniFoldwBkg
from util import read_dict_from_json, get_bins
from plotting import plot_histograms1d, plot_graphs

def load_unfolders(result_dirs, data_sim, vars_mc, vars_det=[], fname_weights='weights.npz', fname_weights_resample='weights_resample25.npz'):
    unfolders = []
    for outdir, data in zip(result_dirs, data_sim):
        # unfolder
        unfolder = OmniFoldwBkg(vars_det, vars_mc)
        unfolder.datahandle_sig = data

        wfiles = []
        wfiles.append(os.path.join(outdir, fname_weights))
        wfiles.append(os.path.join(outdir, fname_weights_resample))
        print('Load event weights from', wfiles)
        unfolder.load(wfiles)

        unfolders.append(unfolder)

    return unfolders

# f_plot
def plot_error_vs_label(unfolders, xlabels, figname, bins, varConfig, **style):
    assert(len(unfolders)==len(xlabels))

    # relative bin errors
    nbins = len(bins)-1
    errors_arr = np.zeros(shape=(len(unfolders), nbins))
    # shape: (n_results, n_bins)

    for i, uf in enumerate(unfolders):
        hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins, normalize=False)[:2]
        errors_arr[i] = hist_err / hist

    # plot graph: error vs label
    plot_graphs(figname, [(xlabels, errors_arr[:,i]) for i in range(nbins)],
                labels=['bin {}'.format(i) for i in range(1, nbins+1)],
                ms=3, ls='-', lw=1, **style)

def plot_error_vs_variable(unfolders, labels, figname, bins, varConfig, **style):
    assert(len(unfolders)==len(labels))

    relerrs_dict = {}
    for i, (uf, label) in enumerate(zip(unfolders, labels)):
        if i==0: # sumw2 error
            hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins, bootstrap_uncertainty=False, normalize=False)[:2]
            relerrs_dict['sumw2'] = hist_err / hist

        hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins, bootstrap_uncertainty=True, normalize=False)[:2]
        relerrs_dict[label] = hist_err / hist

    plot_histograms1d(figname, bins, hists=list(relerrs_dict.values()),
                      labels=list(relerrs_dict.keys()),
                      xlabel=varConfig['xlabel'], ylabel='bin error %',
                      plottypes=['h']*len(relerrs_dict), **style)

def compare(result_dirs, labels, f_plot, plot_label, sim_samples,
            #obs_samples=None,
            outdir='output_compare',
            observables=['mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht', 'th_pt','th_y','th_phi','th_e','tl_pt','tl_y','tl_phi','tl_e'],
            observable_config='configs/observables/vars_klfitter.json',
            bin_config='configs/binning/bins_10equal.json',
            **plot_style):

    # dictionary for observables
    observable_dict = read_dict_from_json(observable_config)

    # variable names at truth level
    vars_mc = [ observable_dict[key]['branch_mc'] for key in observables ]
    # variable names at detector level
    #vars_det = [ observable_dict[key]['branch_det'] for key in observables]
    vars_det = [] # not used

    # load data
    print("Load datasets")
    data_sim = []
    if not isinstance(sim_samples, list) or len(sim_samples)==1:
        # all unfolding results use the same datasets
        dh = DataHandler(sim_samples, variable_names=vars_mc)
        data_sim = [dh] * len(result_dirs)
    elif len(sim_samples)==len(result_dirs):
        for sample, rdir in zip(sim_samples, result_dirs):
            data_sim.append(DataHandler(sample, variable_names=vars_mc))
    else:
        raise RuntimeError("Sample Error")
    #data_truth = DataHandler(obs_samples, variable_names=vars_mc) if obs_samples else None

    # load unfolded weights from all result directories
    print("Load unfolded weights")
    unfolders = load_unfolders(result_dirs, data_sim, vars_mc)

    # output directory
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for varname in observables:
        print(varname)
        varConfig = observable_dict[varname]

        # truth-level bin edges
        bins_mc = get_bins(varname, bin_config)

        # do plot
        figname = os.path.join(outdir, plot_label+'_'+varname)
        f_plot(unfolders, labels, figname, bins_mc, varConfig, title=varname,
               **plot_style)

def get_time_from_log(logfile):
    with open(logfile) as f:
        lines = f.readlines()
        for l in lines:
            if 'Unfolding took' in l:
                words = l.split()
                # time (in seconds) should be the next one after 'took'
                time = words[words.index('took')+1]
                return float(time) # unit: second

    print("No timing information found in ", logfile)
    return None
