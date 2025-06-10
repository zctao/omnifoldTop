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
#run_hist_keywords hist_btag_eigenB bTagSF_DL1r_70_eigenvars_B
#run_hist_keywords hist_btag_eigenC bTagSF_DL1r_70_eigenvars_C
#run_hist_keywords hist_btag_eigenL bTagSF_DL1r_70_eigenvars_Light
#run_hist_keywords hist_theo_scale scale_mu isr_ fsr_
#run_hist_keywords hist_pdf_1 PDF4LHC15_1
#run_hist_keywords hist_pdf_2 PDF4LHC15_2
#run_hist_keywords hist_pdf_3 PDF4LHC15_3 PDF4LHC15_4 PDF4LHC15_5 PDF4LHC15_6 PDF4LHC15_7 PDF4LHC15_8 PDF4LHC15_9
#run_hist_keywords hist_sigmodel hdamp mtop_mt matching_pthard1 ps_hw recoil lineshape_madspin
#run_hist_keywords hist_jet_det JET_EffectiveNP_Detector
#run_hist_keywords hist_jet_mix JET_EffectiveNP_Mixed
#run_hist_keywords hist_jet_mod JET_EffectiveNP_Modelling
#run_hist_keywords hist_jet_stat JET_EffectiveNP_Statistical
#run_hist_keywords hist_jet_eta JET_EtaIntercalibration
#run_hist_keywords hist_jet_flavor JET_Flavor
#run_hist_keywords hist_jet_flav_perjet JET_Flavour_PerJet
#run_hist_keywords hist_jet_jer_datavsmc JET_JER_DataVsMC_MC16
#run_hist_keywords hist_jer_eff_1 JET_JER_EffectiveNP_1_ JET_JER_EffectiveNP_2_ JET_JER_EffectiveNP_3_
#run_hist_keywords hist_jer_eff_2 JET_JER_EffectiveNP_4_ JET_JER_EffectiveNP_5_ JET_JER_EffectiveNP_6_
#run_hist_keywords hist_jer_eff_3 JET_JER_EffectiveNP_7_ JET_JER_EffectiveNP_8_ JET_JER_EffectiveNP_9_
#run_hist_keywords hist_jer_eff_4 JET_JER_EffectiveNP_10_ JET_JER_EffectiveNP_11_ JET_JER_EffectiveNP_12
#run_hist_keywords hist_jet_pileup JET_Pileup
#run_hist_keywords hist_jet_others JET_PunchThrough_MC16 JET_SingleParticle_HighPt

#run_hist_keywords hist_eg EG_RESOLUTION EG_SCALE
#run_hist_keywords hist_met MET_SoftTrk
#run_hist_keywords hist_muon MUON
#run_hist_keywords hist_pu_jvt pileup_DOWN pileup_UP jvt_DOWN jvt_UP

#run_hist_keywords hist_lepsf_el leptonSF_EL_SF
#run_hist_keywords hist_lepsf_mu1 leptonSF_MU_SF_Trigger leptonSF_MU_SF_Isol eptonSF_MU_SF_TTVA
#run_hist_keywords hist_lepsf_mu2 leptonSF_MU_SF_ID

#run_hist_keywords hist_ifsr isr_ fsr_

#run_hist_keywords hist_bkg singleTop_ ttV_ Wjets_ Zjets_ fakes_ VV_
#run_hist_keywords hist_lumi lumi_