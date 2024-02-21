#!/bin/bash
fpath_central="${HOME}/data/OmniFoldOutputs/Run2TTbarXs/Uncertainties/2023Dec06/central/"
fpath_syst="${HOME}/data/OmniFoldOutputs/Run2TTbarXs/Uncertainties/2023Dec06/"
fpath_network="${HOME}/data/OmniFoldOutputs/Run2TTbarXs/Uncertainties/2023Dec06/central/"
output_topdir="${HOME}/data/OmniFoldOutputs/Run2TTbarXs/Uncertainties/2023Dec06/uncertainties"

#groups="JES BTag Lepton MET Pileup IFSR MTop"
groups="JES BTag Lepton MET Pileup IFSR MTop hdamp Hadronization Generator Backgrounds"

#common_args="${fpath_central} -s ${fpath_syst} -t ${fpath_network} -p -v -g ${groups} --observables th_pt th_y tl_pt tl_y mtt ptt ytt ytt_abs th_y_abs"
common_args="${fpath_central} -s ${fpath_syst} -t ${fpath_network} -p -v -g ${groups}"

run_eval() {
    dname="$1"
    shift
    extra_args="$@"

    python scripts/ttbarDiffXsRun2/evaluate_uncertainties.py ${common_args} -o ${output_topdir}/${dname}/ ${extra_args}
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
run_eval abs_trim_p005 --trim-threshold 0.005
# relative
run_eval rel_trim_p001 --normalize --trim-threshold 0.001

### trim and symmetrize
# absolute
run_eval abs_sym_trim_p005 --symmetrize --trim-threshold 0.005
# relative
run_eval rel_sym_trim_p001 --normalize --symmetrize --trim-threshold 0.001