#!/usr/bin/env python3
import os
from compare import plot_error_vs_variable

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

# batch sizes
batch_sizes = [505, 9975]

# error types
#error_types = ['model']

######
# compare bin error components in each bin
print("Plot bin error components")

for ch in ['e', 'm']:
    for bs in batch_sizes:
        print('batch size: {}'.format(bs))
        outdir = os.path.join(plot_dir, 'batchsize{}'.format(bs))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        results = [
            '/home/ztao/work/batch_output/OmniFold/20210224/output_thtlpty_i4_modelerr_bs{}_{}Str_thpt'.format(bs, ch),
            '/home/ztao/work/batch_output/OmniFold/20210227/output_thtlpty_i4_modelerr_bs{}_{}Str_thpt_heinit'.format(bs, ch)
        ]

        plot_error_vs_variable(results, ['Glorot uniform', 'He uniform'],
                                plot_label='unfoldErrors_init_{}'.format(ch),
                                sim_samples = sim_samples[ch], outdir = outdir,
                                observables = observables, plot_sumw2=False)
