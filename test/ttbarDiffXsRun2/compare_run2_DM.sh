#!/bin/bash
###
# OmniFold output
topdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs
timestamp='2024Mar15'
fpath_hist_multifold=${topdir}/Nominal/${timestamp}/nominal/histograms.root
fpath_uncertainties_abs=${topdir}/Uncertainties/uncertainties/abs_sym/bin_uncertainties.root
fpath_uncertainties_rel=${topdir}/Uncertainties/uncertainties/rel_sym/bin_uncertainties.root

###
# Binned
fpath_binned_topdir=${DATA_DIR}/fromDavide/MINI_362_eos
prefix_binned_hist="truth_4j2b_ljets_PseudoTop_Reco"

###
# Make comparison
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/compare_wrt_binned.py \
    ${fpath_hist_multifold} \
    --binned-results-topdir ${fpath_binned_topdir} \
    --binned-histogram-prefix ${prefix_binned_hist} \
    --output-dir ${topdir}/Nominal/${timestamp}/comparison

python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/compare_wrt_binned.py \
    ${fpath_hist_multifold} \
    --binned-results-topdir ${fpath_binned_topdir} \
    --binned-histogram-prefix ${prefix_binned_hist} \
    --output-dir ${topdir}/Nominal/${timestamp}/comparison_werr \
    --fpath-binerrors-absolute ${fpath_uncertainties_abs} \
    --fpath-binerrors-relative ${fpath_uncertainties_rel}