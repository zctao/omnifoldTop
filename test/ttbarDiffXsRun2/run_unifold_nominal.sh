#!/bin/bash
timestamp=${1:-'latest'}
subcampaigns=${2:-'mc16a mc16d mc16e'}

sample_dir=${DATA_DIR}/NtupleTT/20221221
outdir=${DATA_DIR}/OmniFoldOutputs/Run2TTbarXs/NominalUniFold/$timestamp

histogram_topdir=${DATA_DIR}/NtupleTT/20230501/systCRL/ttbar_nominal

observables='mtt ptt th_pt tl_pt ytt th_y tl_y'

#######
echo
echo "Generate run configs"

for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/scripts/ttbarDiffXsRun2/createRun2Config.py \
        --sample-dir ${sample_dir} \
        --result-dir ${outdir}/$obs \
        --config-name ${outdir}/configs/runCfg_${obs} \
        --subcampaigns $subcampaigns \
        --observables $obs \
        --run-list nominal
done

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

echo
echo "Run unfolding"
for obs in ${observables[@]}; do
    echo $obs
    python ${SOURCE_DIR}/run_unfold.py ${outdir}/configs/runCfg_${obs}_nominal.json

    echo
    echo "Make histograms"

    result_dir=${outdir}/$obs/nominal

    python ${SOURCE_DIR}/scripts/make_histograms.py ${result_dir} \
        --binning-config ${SOURCE_DIR}/configs/binning/bins_ttdiffxs.json \
        --observables $obs \
        --binned-correction ${fpath_histogram} --recompute-corrections \
        --include-ibu --compute-metrics -pp -v
        # --binned-noflow
done