#!/usr/bin/env python3
import os
import numpy as np
from datahandler import DataHandler
from omnifoldwbkg import OmniFoldwBkg
from util import read_dict_from_json, get_bins
from plotting import plot_histograms1d, plot_graphs

def load_unfolders(result_dirs, data_sim, vars_mc, vars_det=[]):
    unfolders = []
    for outdir in result_dirs:
        # unfolder
        unfolder = OmniFoldwBkg(vars_det, vars_mc)
        unfolder.datahandle_sig = data_sim

        wfiles = []
        wfiles.append(os.path.join(outdir, 'weights.npz'))
        wfiles.append(os.path.join(outdir, 'weights_resample25.npz'))
        unfolder.load(wfiles)

        unfolders.append(unfolder)

    return unfolders

obs_samples = [
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_1_klcut.npz']
sim_samples = [
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_2_klcut.npz']

# observables
observables = [#'Ht'
    'mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht',
    'th_pt','th_y','th_phi','th_e','tl_pt','tl_y','tl_phi','tl_e'
]

# dictionary for observables
observable_dict = read_dict_from_json('configs/observables/vars_klfitter.json')

# binning configuration
bin_config = 'configs/binning/bins_10equal.json'

# variable names at truth level
vars_mc = [ observable_dict[key]['branch_mc'] for key in observables ]
# variable names at detector level
vars_det = [] # not used

# load data
print("Loading datasets")
data_sim = DataHandler(sim_samples, variable_names=vars_mc)
#data_truth = DataHandler(obs_samples, variable_names=vars_mc)

error_types = ['full', 'stat', 'model']
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
markers = ['o', '*', '+', 's', 'd', 'x', 'h', '^', 'v', 'p']

# output directory
plotdir = 'output_compare'
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

######
# plot errors as a function of batch size
print('Plot errors as a function of batch size')

for etype in error_types:
    break
    print(etype)
    result_dirs = []
    for bs in batch_sizes:
        result_dirs.append('output_err_batc{}_{}_eClos'.format(bs, etype))

    unfolders = load_unfolders(result_dirs, data_sim, vars_mc)

    for varname in observables:
        varConfig = observable_dict[varname]

        # bin edges for truth-level distribution
        bins_mc = get_bins(varname, bin_config)

        # bin errors
        nbins = len(bins_mc)-1
        errors_arr = np.zeros(shape=(len(batch_sizes), nbins))
        # shape: (n_results, n_bins)

        for ir, uf in enumerate(unfolders):
            hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins_mc)[:2]
            errors_arr[ir] = hist_err / hist

        plot_graphs(plotdir+'/binerrors_{}_{}'.format(etype, varname),
                    [(np.array(batch_sizes), errors_arr[:,i]) for i in range(nbins)],
                    labels=['bin {}'.format(i) for i in range(1, nbins+1)],
                    xlabel='training batch size',
                    ylabel='bin error ({}) %'.format(etype), ms=3,
                    title=varname, xscale='log2', markers=markers, ls='-', lw=1)

######
# compare different types of errors in each bin
print("Plot different components of bin errors")

for bs in batch_sizes:
    print('batch size: {}'.format(bs))
    plotdir = os.path.join('output_compare', 'batchsize{}'.format(bs))
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    result_dirs = []
    for etype in error_types:
        result_dirs.append('output_err_batc{}_{}_eClos'.format(bs, etype))

    unfolders = load_unfolders(result_dirs, data_sim, vars_mc)

    for varname in observables:
        varConfig = observable_dict[varname]

        # bin edges for truth-level distribution
        bins_mc = get_bins(varname, bin_config)

        relerrs_dict = {}
        for ir, (uf, etype) in enumerate(zip(unfolders, error_types)):
            if ir==0: # sumw2 error
                hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins_mc, bootstrap_uncertainty=False)[:2]
                relerrs_dict['sumw2'] = hist_err / hist

            hist, hist_err = uf.get_unfolded_distribution(varConfig['branch_mc'], bins_mc, bootstrap_uncertainty=True)[:2]
            relerrs_dict[etype] = hist_err / hist

        plot_histograms1d(plotdir+'/unfoldErrors_'+varname, bins_mc,
                        hists=list(relerrs_dict.values()),
                        labels=list(relerrs_dict.keys()),
                        title=varname, xlabel=varConfig['xlabel'],
                        ylabel='bin error %',
                        plottypes=['h']*len(relerrs_dict))
