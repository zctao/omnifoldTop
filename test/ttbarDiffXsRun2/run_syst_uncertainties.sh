#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/Uncertainties/$timestamp

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
#observables_multidim='ptt_vs_mtt th_pt_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

######
echo
echo "Generate run configs"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir} \
    --config-name ${outdir}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list systematics \
    --config-string '{"binning_config":"${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json"}' # explicitly specify binning config so that histograms are made right after unfolding
    # -k bTagSF_DL1r_70_eigenvars_B1 CategoryReduction_JET_Pileup_RhoTopology

######
echo
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_syst.json

# for computing percentage bin errors, no need to remake histograms with acceptance and efficiency corrections

#####
echo
echo "Evaluate uncertainties"
echo "Absolute"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/evaluate_uncertainties.py \
    ${outdir}/central \
    -s ${outdir} \
    -o ${outdir}/uncertainties/abs \
    -p
    # -k bTagSF_DL1r_70_eigenvars_B1 CategoryReduction_JET_Pileup_RhoTopology

echo "Relative"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/evaluate_uncertainties.py \
    ${outdir}/central \
    -s ${outdir} \
    -o ${outdir}/uncertainties/rel --normalize \
    -p