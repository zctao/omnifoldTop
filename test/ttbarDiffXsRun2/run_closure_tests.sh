#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
output_dir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/ClosureTests/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

######
echo "Generate run configs for closure tests"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${output_dir} \
    --config-name ${output_dir}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list closure 

######
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${output_dir}/configs/runCfg_closure_oddeven.json
python ${SOURCE_DIR}/run_unfold.py ${output_dir}/configs/runCfg_closure_resample.json