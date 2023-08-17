#!/bin/bash
result_top_dir=${DATA_DIR}/OmniFoldOutputs/Run2
samples_dir=${DATA_DIR}/NtupleTT/latest

python scripts/ttbarDiffXsRun2/createRun2Config.py \
    -d ${samples_dir} \
    -r ${result_top_dir} \
    -n ${result_top_dir}/configs/runCfg_run2_ljets \
    -e mc16a mc16d mc16e \
    -b \
    -s #-k bTagSF_DL1r_70_eigenvars_B1 CategoryReduction_JET_Pileup_RhoTopology
    #--observables th_pt