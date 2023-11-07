#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir_rw=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Reweight/$timestamp
outdir_test=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/StressTests/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

#######
echo "Generate pseudo data by reweighting signal MC to data"

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/reweightDataStress.py \
    ${sample_dir} ${outdir_rw}/nnrw \
    --observables $observables \
    -e $subcampaigns

fpath_reweights=${outdir_rw}/nnrw/reweights.h5

######
echo "Generate run configs for stress tests"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir_test} \
    --config-name ${outdir_test}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list nominal stress \
    --external-reweights ${fpath_reweights}

######
echo "Run unfolding"
cfg_suffix=(stress_data stress_data_alt nominal stress_th_pt stress_bump)
for sfx in ${cfg_suffix[@]}; do
    echo $sfx
    python ${SOURCE_DIR}/run_unfold.py ${outdir_test}/configs/runCfg_${sfx}.json
done