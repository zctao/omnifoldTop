#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir_rw=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Reweight/$timestamp
outdir_test=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/StressTests/$timestamp

observables_stress='th_pt mtt ptt'
observables_test='mtt ptt th_pt tl_pt ytt th_y tl_y'

echo "Generate pseudo data from binned reweighting"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/reweightDataStressBinned.py \
    ${sample_dir} ${outdir_rw}/binrw \
    --observables ${observables_stress} \
    -e $subcampaigns

fpath_reweights_binned=${outdir_rw}/binrw

for obs_s in ${observables_stress[@]}; do
    ######
    echo "Generate run configs for stress tests (binned) on ${obs_s}"

    fpath_reweights_binned=${outdir_rw}/binrw/${obs_s}/reweights.h5

    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir_test}/stress_data_binned_${obs_s} \
        --config-name ${outdir_test}/configs/runCfg_${obs_s} \
        --subcampaigns $subcampaigns \
        --observables $observables_test \
        --run-list stress_binned \
        --external-reweights ${fpath_reweights_binned} #--config-string '{"match_dR":0.8}'

    ######
    echo "Run unfolding"
    python ${SOURCE_DIR}/run_unfold.py ${outdir_test}/configs/runCfg_${obs_s}_stress_data_binned.json

done