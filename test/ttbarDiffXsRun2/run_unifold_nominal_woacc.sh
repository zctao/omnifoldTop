#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/NominalUniFold_wacc/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

#######
echo
echo "Generate run configs"

for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir}/$obs \
        --config-name ${outdir}/configs/runCfg_woacc_${obs} \
        --subcampaigns $subcampaigns \
        --observables $obs \
        --run-list nominal \
        --config-string '{"correct_acceptance":false}'
done

echo
echo "Run unfolding"
for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_woacc_${obs}_nominal.json

    echo
    echo "Make histograms"

    result_dir=${outdir}/$obs/nominal

    python ${SOURCE_DIR}/scripts/make_histograms.py ${result_dir} \
        --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
        --observables $obs \
        --include-ibu --compute-metrics -pp -v
        # --binned-noflow
done