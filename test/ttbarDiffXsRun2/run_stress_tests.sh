#!/bin/bash
timestamp='2023Aug17'
subcampaigns='mc16a mc16d mc16e'

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir_rw=${DATA_DIR}/OmniFoldOutputs/Reweight/$timestamp
outdir_test=${DATA_DIR}/OmniFoldOutputs/StressTests/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

#######
echo "Generate pseudo data by reweighting signal MC to data"

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/reweightDataStress.py \
    ${sample_dir} ${outdir_rw}/run2 \
    --observables $observables \
    -e $subcampaigns

fpath_reweights=${outdir_rw}/run2/reweights.h5

# mc16a only
#python $SOURCE_DIR/scripts/ttbarDiffXsRun2/reweightDataStress.py \
#    ${sample_dir} ${output_dir}/mc16a \
#    --observables mtt ptt th_pt tl_pt ytt th_y tl_y \
#    -e mc16a

#fpath_reweights=${outdir_rw}/mc16a/reweights.h5

######
echo "Generate run configs for stress tests"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir_test} \
    --config-name ${outdir_test}/configs/runCfg_run2_ljets \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list nominal stress \
    --external-reweights ${fpath_reweights}

######
echo "Run unfolding"
cfg_suffix=(stress_data stress_data_alt nominal stress_th_pt stress_bump)
for sfx in ${cfg_suffix[@]}; do
    echo $sfx
    python ${SOURCE_DIR}/run_unfold.py ${outdir_test}/configs/runCfg_run2_ljets_${sfx}.json
done