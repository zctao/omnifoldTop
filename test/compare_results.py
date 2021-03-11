#!/usr/bin/env python3
import os
from compare import plot_error_vs_label, plot_error_vs_variable
from compare import get_time_from_log
from plotting import plot_graphs

# observables to plot
observables = ['th_pt', 'th_y', 'tl_pt', 'tl_y']

# file names of the unfolded weights
fname_weights = 'weights.npz'
fname_weights_resample='weights_resample25.npz'

# samples
obs_samples = {
    'e': ['/home/ztao/work/data/TopNtupleAnalysis/mc16e/21_2_147/20210216/ttbar/re_ttbar_0_klcut.npz'],
    'm': ['/home/ztao/work/data/TopNtupleAnalysis/mc16e/21_2_147/20210216/ttbar/rmu_ttbar_0_klcut.npz']
}

sim_samples = {
    'e': ['/home/ztao/work/data/TopNtupleAnalysis/mc16e/21_2_147/20210216/ttbar/re_ttbar_1_klcut.npz'],
    'm': ['/home/ztao/work/data/TopNtupleAnalysis/mc16e/21_2_147/20210216/ttbar/rmu_ttbar_1_klcut.npz']
}

# plot directory
plot_dir = '/home/ztao/work/batch_output/OmniFold/compare_batchsize'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# unfolding results
result_dir = '/home/ztao/work/batch_output/OmniFold/20210224/output_thtlpty_i4_{}err_bs{}_{}Str_thpt'

# batch sizes
batch_sizes = [47, 101, 252, 505, 950, 2520, 5000, 9975, 20000]

# error types
error_types = ['model'] #['model', 'full']

# marker styles for 10 bins
markers = ['o', '*', '+', 's', 'd', 'x', 'h', '^', 'v', 'p']

######
# plot bin errors vs batch size
print("Plot errors as a function of batch sizes")
for ch in ['e', 'm']:
    for errtype in error_types:
        print(ch, errtype)
        results = []
        labels = []
        for bs in batch_sizes:
            results.append(result_dir.format(errtype, bs, ch))
            labels.append(bs)

        plot_error_vs_label(results, labels,
                            plot_label = 'binerrors_{}_{}'.format(ch, errtype),
                            sim_samples = sim_samples[ch], outdir = plot_dir,
                            observables = observables, markers=markers,
                            xlabel='batch size', xscale='log2',
                            ylabel='relative bin error ({})'.format(errtype))

######
# plot training time vs batch size
print("Plot training time as a function of batch sizes")
for ch in ['e', 'm']:
    for errtype in error_types:
        print(ch, errtype)
        results = []
        labels = []
        times = []
        for bs in batch_sizes:
            logfile = result_dir.format(errtype, bs, ch) + '/log.txt'
            t = get_time_from_log(logfile) / 60. # minutes
            t = t / (1+25) # divided by one nominal run and 25 resamples
            times.append(t)
            labels.append(bs)

        # plot
        figname = os.path.join(plot_dir, 'time_{}_{}'.format(ch, errtype))
        print("Create plot: {}".format(figname))
        plot_graphs(figname, [(labels, times)], title='Unfolding time', xlabel='batch size', xscale='log2', ylabel='minutes', markers=['o'])

######
# compare bin error components in each bin
print("Plot bin error components")

# batch sizes
#batch_sizes = [505]

# error types
#error_types = ['model', 'full']

for ch in ['e', 'm']:
    for bs in batch_sizes:
        print('batch size: {}'.format(bs))
        outdir = os.path.join(plot_dir, 'batchsize{}'.format(bs))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        results = []
        for errtype in error_types:
            results.append(result_dir.format(errtype, bs, ch))

        plot_error_vs_variable(results, error_types,
                                plot_label='unfoldErrors_{}'.format(ch),
                                sim_samples = sim_samples[ch], outdir = outdir,
                                observables = observables)
            
