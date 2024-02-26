#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs//Run2TTbarXs/Uncertainties/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
observables_multidim='ptt_vs_mtt th_pt_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

######
# Generate run configs for bootstraping
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir} \
    --config-name ${outdir}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list bootstrap

######
# Run unfolding
python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_bootstrap.json
python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_bootstrap_mc.json

######
# Make histograms
bs_dir=${outdir}/bootstrap
for i in {0..9}; do
    python ${SOURCE_DIR}/scripts/make_histograms.py ${bs_dir}/resamples${i} \
        --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
        --observables $observables \
        --observables-multidim ${observables_multidim} \
        --include-ibu \
        -v
done

bs_mc_dir=${outdir}/bootstrap_mc
for i in {0..9}; do
    python ${SOURCE_DIR}/scripts/make_histograms.py ${bs_mc_dir}/resamples${i} \
        --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
        --observables $observables \
        --observables-multidim ${observables_multidim} \
        --include-ibu \
        -v
done

######
# Evaluate by scripts/ttbarDiffXsRun2/evaluate_uncertainties.py together with systematic uncertainties