#!/usr/bin/env python3
import os
from compare import get_time_from_log
from plotting import plot_graphs

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
error_types = ['full', 'model']

plotdir='output_compare_freenorm'
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)

for etype in error_types:
    print(etype)

    results_dir = []
    labels = []
    times = []

    for bs in batch_sizes:
        result_dir = 'output_err_batc{}_{}_eClos'.format(bs, etype)
        logfile = result_dir + '/log.txt'
        t = get_time_from_log(logfile) / 60. # minutes
        times.append(t)
        labels.append('{}'.format(bs))

    # plot
    figname = os.path.join(plotdir, 'timing_{}'.format(etype))
    print("Create plot: {}".format(figname))
    plot_graphs(figname, [(labels, times)], title='Unfolding time', xlabel='batch size', ylabel='minutes', markers=['o'])

###
result_dirs = ['output_1d_bs512_full_eClos', 'output_2d_bs1024_full_eClos', 'output_4d_bs2048_full_eClos']
labels = ['batch size 512\nx1 data', 'batch size 1024\nx2 data', 'batch size 2048\nx4 data']
times = []
for rdir in result_dirs:
    logfile = rdir + '/log.txt'
    t = get_time_from_log(logfile) / 60. # minutes
    times.append(t)

figname = os.path.join(plotdir, 'timing_eqnbatch_full')
print("Create plot: {}".format(figname))
plot_graphs(figname, [(labels, times)], title='Unfolding time', ylabel='minutes', markers=['o'])

###
result_dirs = ['output_1d_bs512_full_eClos','output_2d_bs512_full_eClos','output_3d_bs512_full_eClos']
labels = ['x1 data', 'x2 data', 'x3 data']
times = []
for rdir in result_dirs:
    logfile = rdir + '/log.txt'
    t = get_time_from_log(logfile) / 60. # minutes
    times.append(t)

figname = os.path.join(plotdir, 'timing_bs512_ndatasets_full')
print("Create plot: {}".format(figname))
plot_graphs(figname, [(labels, times)], title='Unfolding time', xlabel='batch size 512', ylabel='minutes', markers=['o'])
