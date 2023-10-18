#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns='mc16a'

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir_test=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/GridSearch/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

# Uncomment the following if wish to rerun reweighting
#outdir_rw=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Reweight/$timestamp

#python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/reweightDataStress.py \
#    ${sample_dir} ${outdir_rw}/nnrw \
#    --observables $observables \
#    -e $subcampaigns
#
#fpath_reweights=${outdir_rw}/nnrw/reweights.h5

# or use the previous results
fpath_reweights=${DATA_DIR}/OmniFoldOutputs/Reweight/2023Aug24_mc16a/nnrw/reweights.h5

# networks to scan
networks="dense_100x3 dense_50x3 dense_10x3 dense_100x1 dense_1000x3 dense_100x10 dense_10x1 dense_50x1 dense_1000x1 dense_10x10 dense_50x10 dense_1000x10"
#echo $networks

model_names=''
for mn in ${networks[@]}; do
    model_names=${model_names}'"'${mn}'":"'${mn}'",'
done
#echo ${model_names}

######
echo "Generate run configs"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir_test} \
    --config-name ${outdir_test}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list stress \
    --external-reweights ${fpath_reweights} \
    --config-string '{"model_name": {'${model_names}'}, "match_dR":0.8, "run_ibu":false, "iterations":5, "nruns":5, "parallel_models":5}'

######
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${outdir_test}/configs/runCfg_stress_data.json

######
echo "Compare performance"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/evaluateGridSearch.py \
    ${outdir_test} \
    ${networks} \
    --iterations 3 4 5 best \
    -o ${outdir_test}/plots