#!/bin/bash

for lr in 0.01 0.001 0.0001 0.00001; do
    output="/data/wcassidy/output/learning_rate/closure/$lr/"
    mkdir -p "$output"
    cmd=$(cat <<EOF
cd ~/omnifoldTop
source ~/omnifoldTop/setup.sh
OF_LR=$lr python ~/omnifoldTop/unfold.py \
  -d /data/wcassidy/ttbar/sim/mntuple_ttbar_[012]_parton_ejets_matched.npz \
  -s /data/wcassidy/ttbar/sim/mntuple_ttbar_6_parton_ejets_matched.npz \
  -o $output \
  --nresamples 10 \
  -e bootstrap_model \
  --truth-known \
  --plot-correlations \
  --plot-history
EOF
    )
    singularity exec --nv --bind /data ~/omnifoldTop/baseml_tf_v0.1.33.sif bash <<EOF || $(error="$?"; printf "Failed in $lr\n" >&2; exit "$error") || exit $?
$cmd
EOF
    printf "$cmd\n" > "$output"/invocation.txt
done
