#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/AccCorrValidation/$timestamp

histogram_topdir=${DATA_DIR}/NtupleTT/20240308/systCRL/ttbar_nominal

#######
echo "Merge histograms for binned corrections"
fpath_histogram=${outdir}/histograms_merged.root
if [ -f ${fpath_histogram} ]; then
    echo "Read histograms from ${fpath_histogram}"
else
    echo "Merge histogram files"

    declare -a sub_arr=($subcampaigns)
    histdir_str=''
    for sub in ${sub_arr[@]}; do
        histdir_str=${histdir_str}${histogram_topdir}/$sub' '
    done

    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/collectHistograms.py ${histdir_str} -o ${fpath_histogram} -v
fi

run_multidim() {
    local label=$1
    shift 1
    local obs=$@

    echo $label
    echo "Generate run config with acceptance corrections"
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir}/${label}_wacc \
        --config-name ${outdir}/configs/runCfg_${label}_wacc \
        --subcampaigns $subcampaigns \
        --observables $obs \
        --run-list nominal \
        --config-string '{"correct_acceptance":true, "match_dR":0.8,"run_ibu":false}'

    echo "Generate run config without acceptance corrections"
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir}/${label}_woacc \
        --config-name ${outdir}/configs/runCfg_${label}_woacc \
        --subcampaigns $subcampaigns \
        --observables $obs \
        --run-list nominal \
        --config-string '{"correct_acceptance":false, "match_dR":0.8,"run_ibu":false}'

    echo "Run unfolding and make histograms"
    python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_${label}_wacc_nominal.json
    python ${SOURCE_DIR}/scripts/make_histograms.py ${outdir}/${label}_wacc/nominal \
        --observables $obs \
        --observables-multidim $label \
        --binned-correction ${fpath_histogram} --recompute-corrections \
        --include-ibu -v

    python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_${label}_woacc_nominal.json
    python ${SOURCE_DIR}/scripts/make_histograms.py ${outdir}/${label}_woacc/nominal \
        --observables $obs \
        --observables-multidim $label \
        --binned-correction ${fpath_histogram} --recompute-corrections \
        --include-ibu -v

    # compare the ratios of the unfolded histogram with and without acceptance corrections between UniFold and IBU
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/compareAcceptance.py \
        ${outdir}/${label}_wacc/nominal/histograms.root \
        ${outdir}/${label}_woacc/nominal/histograms.root \
        -o ${outdir}/plots/${label}
}

run_multidim ptt_vs_mtt ptt mtt
run_multidim th_pt_vs_mtt th_pt mtt
run_multidim ytt_abs_vs_mtt ytt_abs mtt
run_multidim ptt_vs_ytt_abs ptt ytt_abs
run_multidim mtt_vs_ytt_abs mtt ytt_abs