#!/usr/bin/env python3
import os
from compare import compare, plot_error_vs_label, plot_error_vs_variable

obs_samples = [
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_1_klcut.npz']
sim_samples = [
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_2_klcut.npz']

# observables
observables = [#'Ht'
    'mtt', 'ptt', 'ytt', 'ystar', 'chitt', 'yboost', 'dphi', 'Ht',
    'th_pt','th_y','th_phi','th_e','tl_pt','tl_y','tl_phi','tl_e'
]

# batch sizes
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
markers = ['o', '*', '+', 's', 'd', 'x', 'h', '^', 'v', 'p']

# error type: full, model
error_types = ['full', 'model']

##########################
# error vs batch size
print('Plot errors as a function of batch size')
plotdir='output_compare_freenorm'
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

for etype in error_types:
    print(etype)
    result_dirs = []
    labels = []
    for bs in batch_sizes:
        result_dirs.append('output_err_batc{}_{}_eClos'.format(bs, etype))
        labels.append('{}'.format(bs))

    compare(result_dirs, labels, plot_error_vs_label, sim_samples=sim_samples,
            outdir=plotdir, plot_label='binerrors_{}'.format(etype), 
            observables=observables, markers=markers,
            xlabel='batch size', ylabel='bin error ({}) %'.format(etype))

# batch size 512
sim_samples_nbatch = [
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_1_klcut.npz',
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_3_klcut.npz',
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_5_klcut.npz',
    '/data/ztao/TopNtupleAnalysis/mc16e/21_2_147/ttbar/re_ttbar_7_klcut.npz']

result_dirs = ['output_1d_bs512_full_eClos', 'output_2d_bs1024_full_eClos', 'output_4d_bs2048_full_eClos']
labels = ['batch size 512\nx1 data', 'batch size 1024\nx2 data', 'batch size 2048\nx4 data']
sample_list= [sim_samples_nbatch[:1], sim_samples_nbatch[:2], sim_samples_nbatch[:4]]

compare(result_dirs, labels, plot_error_vs_label,
        sim_samples=sample_list,
        outdir=plotdir, plot_label='binerrors_eqnbatch_full',
        observables=observables)

result_dirs = ['output_1d_bs512_full_eClos','output_2d_bs512_full_eClos','output_3d_bs512_full_eClos']
labels = ['x1 data', 'x2 data', 'x3 data']
sample_list=[sim_samples_nbatch[:1], sim_samples_nbatch[:2], sim_samples_nbatch[:3]]

compare(result_dirs, labels, plot_error_vs_label, sim_samples=sample_list,
        outdir=plotdir, plot_label='binerrors_bs512_ndatasets_full',
        observables=observables, xlabel='batch size 512')

##########################
# compare different types of errors in each bin
print("Plot bin error components")
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
error_types = ['model', 'full']

for bs in batch_sizes:
    print('batch size: {}'.format(bs))
    plotdir = os.path.join('output_compare_freenorm', 'batchsize{}'.format(bs))
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    result_dirs = []
    for etype in error_types:
        result_dirs.append('output_err_batc{}_{}_eClos'.format(bs, etype))

    compare(result_dirs, error_types, plot_error_vs_variable,
            outdir=plotdir, plot_label='unfoldErrors', sim_samples=sim_samples,
            observables=observables)
