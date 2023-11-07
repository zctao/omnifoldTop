#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir_rw=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Reweight/$timestamp
outdir_test=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/StressTests/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

echo "Generate pseudo data from binned reweighting"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/reweightDataStressBinned.py \
    ${sample_dir} ${outdir_rw}/binrw \
    --observables $observables \
    -e $subcampaigns

fpath_reweights_binned=${outdir_rw}/binrw

######
echo "Generate run configs for stress tests (binned)"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir_test} \
    --config-name ${outdir_test}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list stress_binned \
    --external-reweights ${fpath_reweights_binned}

######
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${outdir_test}/configs/runCfg_stress_data_binned.json