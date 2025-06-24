#!/bin/bash
TOPDIR="${HOME}/data/OmniFoldOutputs/Run2TTbarXs_MINI382"

fpath_syst="${TOPDIR}/Uncertainties/20250216"
fpath_central="${fpath_syst}/central"
fpath_network="${fpath_syst}/central"
fpath_stress="${TOPDIR}/StressTests/20250321/stress_data"
fpath_stat="${TOPDIR}/Uncertainties/20250506/bootstrap"
fpath_stat_mc="${TOPDIR}/Uncertainties/20250506/bootstrap_mc"

output_topdir="${TOPDIR}/Uncertainties/binned_uncertainties"

groups="JES BTag Lepton+MET Pileup Backgrounds Modelling IFSR PDF MTop hdamp"

run_eval() {
    dname="$1"
    shift
    extra_args="$@"

    python scripts/ttbarDiffXsRun2/evaluate_uncertainties.py \
        ${fpath_central} \
        -s ${fpath_syst} \
        -t ${fpath_network} \
        -u ${fpath_stress} \
        -g ${groups} \
        -p -v \
        -o ${output_topdir}/${dname} ${extra_args} \
        -b ${fpath_stat} \
        -m ${fpath_stat_mc}
}

# absolute
run_eval abs
# relative
run_eval rel --normalize

### symmetrize
# absolute
run_eval abs_sym --symmetrize
# relative
run_eval rel_sym --normalize --symmetrize

### trim
#run_eval abs_trim_p005 --trim-threshold 0.005
# relative
#run_eval rel_trim_p001 --normalize --trim-threshold 0.001

### trim and symmetrize
# absolute
#run_eval abs_sym_trim_p005 --symmetrize --trim-threshold 0.005
# relative
#run_eval rel_sym_trim_p001 --normalize --symmetrize --trim-threshold 0.001