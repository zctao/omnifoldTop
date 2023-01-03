#!/bin/bash

result_top_dir=${1:-${HOME}/data/OmniFoldOutputs/Run2}
samples_dir=${2:-${HOME}/atlasserv/NtupleTT/latest}

for ch in ljets; do  # or: ejets mjets
    python scripts/ttbarDiffXsRun2createRun2Config.py \
           -d ${samples_dir} \
           -c ${ch} \
           -r ${result_top_dir} \
           -n configs/run/ttbarDiffXsRun2/runCfg_run2_${ch} \
           -e mc16a mc16d mc16e \
           -b \
           -s all
           #-s CategoryReduction_JET_Pileup_RhoTopology
done
