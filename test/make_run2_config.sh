#!/bin/bash

result_top_dir=${1}
samples_dir=${2:-${HOME}/data/ttbarDiffXs13TeV/latest}

for ch in ejets mjets; do
    python scripts/createRun2Config.py \
           -d ${samples_dir} \
           -c ${ch} \
           -r ${result_top_dir} \
           -n configs/run/runCfg_run2_${ch} \
           -e mc16a mc16d mc16e \
           -b \
           -s all
           #-s CategoryReduction_JET_Pileup_RhoTopology
done
