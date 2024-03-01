#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs//Run2TTbarXs/Nominal_woacc/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
observables_multidim='ptt_vs_mtt th_pt_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

#######
echo
echo "Generate run configs"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir} \
    --config-name ${outdir}/configs/runCfg_woacc \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list nominal \
    --config-string '{"correct_acceptance":false}'

######
echo
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_woacc_nominal.json

######
echo
echo "Make histograms"
result_dir=${outdir}/nominal

python ${SOURCE_DIR}/scripts/make_histograms.py ${result_dir} \
    --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
    --observables $observables \
    --include-ibu --compute-metrics -pp -v \
    --observables-multidim ${observables_multidim}
    # --binned-noflow