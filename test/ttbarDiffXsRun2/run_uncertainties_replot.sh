#!/bin/bash

uncertainty_dir="${HOME}/data/OmniFoldOutputs/Run2TTbarXs/Uncertainties/2023Dec06/uncertainties"
uncertainty_fname="bin_uncertainties.root"

unc_types=( abs rel abs_sym rel_sym abs_trim_p005 rel_trim_p001 abs_sym_trim_p005 rel_sym_trim_p001 )

for utyp in "${unc_types[@]}"; do
    python scripts/ttbarDiffXsRun2/plot_uncertainties.py ${uncertainty_dir}/${utyp}/${uncertainty_fname} -a
done