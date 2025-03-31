#!/bin/bash
syst_topdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs_MINI382/Uncertainties/20250216/

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
observables_multidim='ptt_vs_mtt th_pt_vs_mtt ytt_abs_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

run_hist_group() {
    jobname="$1"
    shift
    groups="$@"

    python scripts/ttbarDiffXsRun2/makehist_uncertainties.py \
        ${jobname} -g ${groups} \
        -s ${syst_topdir} -j ${syst_topdir}/slurm_histjobs \
        --observables ${observables} ${observables_multidim} \
        -v # -b --rerun
}

run_hist_keywords() {
    jobname="$1"
    shift
    keywords="$@"

    python scripts/ttbarDiffXsRun2/makehist_uncertainties.py \
        ${jobname} -k ${keywords} \
        -s ${syst_topdir} -j ${syst_topdir}/slurm_histjobs \
        --observables ${observables} ${observables_multidim} \
        -v # -b --rerun
}

#Examples:
#run_hist_group hist_btag BTag
#or:
#run_hist_keywords hist_btag_extrap bTagSF_DL1r_70_extrapolation