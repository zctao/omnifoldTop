#!/bin/bash
histograms_dir=${1:-${DATA_DIR}/NtupleTT/20230417/systCRL/ttbar_nominal}

python scripts/ttbarDiffXsRun2/binnedCorrections.py ${histograms_dir} \
    --output ${histograms_dir}/ttbar_binned_corrections.root \
    --histogram-outname ${histograms_dir}/ttbar_histograms_run2.root \
    -v #--observables th_pt mtt ptt_vs_mtt mtt_vs_ptt_vs_ytt_abs

python scripts/ttbarDiffXsRun2/binnedCorrections.py \
    ${histograms_dir}/ttbar_histograms_run2.root \
    --no-flow \
    --output ${histograms_dir}/ttbar_binned_corrections_noflow.root \
    -v #--observables th_pt mtt ptt_vs_mtt mtt_vs_ptt_vs_ytt_abs

python scripts/ttbarDiffXsRun2/binnedCorrections.py \
    ${histograms_dir}/ttbar_histograms_run2.root \
    --mc-weight \
    --output ${histograms_dir}/ttbar_binned_corrections_mcw.root \
    -v #--observables th_pt mtt ptt_vs_mtt mtt_vs_ptt_vs_ytt_abs

python scripts/ttbarDiffXsRun2/binnedCorrections.py \
    ${histograms_dir}/ttbar_histograms_run2.root \
    --no-flow \
    --mc-weight \
    --output ${histograms_dir}/ttbar_binned_corrections_noflow_mcw.root \
    -v #--observables th_pt mtt ptt_vs_mtt mtt_vs_ptt_vs_ytt_abs