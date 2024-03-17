#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs//Run2TTbarXs/Nominal/$timestamp

histogram_topdir=${DATA_DIR}/NtupleTT/20240308/systCRL/ttbar_nominal

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'
observables_multidim='ptt_vs_mtt th_pt_vs_mtt ytt_abs_vs_mtt ptt_vs_ytt_abs mtt_vs_ytt_abs mtt_vs_ptt_vs_ytt_abs mtt_vs_th_pt_vs_th_y_abs mtt_vs_th_pt_vs_ytt_abs mtt_vs_th_y_abs_vs_ytt_abs'

#######
echo
echo "Generate run configs"
python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
    --sample-dir ${sample_dir} \
    --result-dir ${outdir} \
    --config-name ${outdir}/configs/runCfg \
    --subcampaigns $subcampaigns \
    --observables $observables \
    --run-list nominal

######
echo
echo "Run unfolding"
python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_nominal.json

######
# Binned corrections
echo
fpath_histogram=${outdir}/histograms_merged.root
if [ -f ${fpath_histogram} ]; then
    echo "Read histograms from ${fpath_histogram}"
else
    echo "Merge histogram files"

    declare -a sub_arr=($subcampaigns)
    histdir_str=''
    for sub in ${sub_arr[@]}; do
        #echo ${histogram_topdir}/$sub
        histdir_str=${histdir_str}${histogram_topdir}/$sub' '
    done

    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/collectHistograms.py ${histdir_str} -o ${fpath_histogram} -v
fi

# Computation of the binned corrections is actually done in make_histograms.py below using the merged histograms

######
echo
echo "Make histograms"
result_dir=${outdir}/nominal

python ${SOURCE_DIR}/scripts/make_histograms.py ${result_dir} \
    --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
    --observables $observables \
    --binned-correction ${fpath_histogram} --recompute-corrections \
    --include-ibu --compute-metrics -pp -v \
    --observables-multidim ${observables_multidim}
    # --binned-noflow