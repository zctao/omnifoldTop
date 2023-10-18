#!/bin/bash
timestamp=${1:-'latest'}
topdir=${DATA_DIR}/OmniFoldOutputs//Run2TTbarXs
fpath_nomimal=${topdir}/Nominal/$timestamp/nominal/histograms.root
fpath_uncertainties_abs=${topdir}/Uncertainties/$timestamp/uncertainties/abs/bin_uncertainties.root
fpath_uncertainties_rel=${topdir}/Uncertainties/$timestamp/uncertainties/rel/bin_uncertainties.root
fpaths_otherMCs=${fpath_nomimal}

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/plotDiffXs.py \
    ${fpath_nomimal} \
    -a ${fpath_uncertainties_abs} \
    -r ${fpath_uncertainties_rel} \
    -o ${topdir}/results \
    --observables th_pt ptt_vs_mtt mtt_vs_th_y_abs_vs_ytt_abs \
    --label-nominal PWG+PY8 \
    -v
    #-m ${fpaths_otherMCs} \
    #--labels-otherMC OtherTest