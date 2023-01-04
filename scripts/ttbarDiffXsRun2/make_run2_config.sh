#!/bin/bash
result_top_dir=/mnt/xrootdg/ztao/OmniFoldOutputs/Run2
samples_dir=/mnt/xrootdg/ztao/NtupleTT/latest

python scripts/ttbarDiffXsRun2/createRun2Config.py \
    -d ${samples_dir} \
    -r ${result_top_dir} \
    -n configs/run/ttbarDiffXsRun2/runCfg_run2_ljets \
    -e mc16a mc16d mc16e \
    -b \
    -s all
    #-s CategoryReduction_JET_Pileup_RhoTopology
    #-s pileup
    #-s bTagSF_DL1r_70_eigenvars_B:0