#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/ntuplerTT/latest
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs_MINI382/Uncertainties/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

echo "Generate unfolding configs"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/run_uncertainties.py \
    -r ${outdir} \
    --rerun -v \
    generate \
    --sample-dir ${sample_dir} \
    --observables ${observables} \
    -e ${subcampaigns} \
    -b cc ubc # --config-string '{"match_dR":0.8}'