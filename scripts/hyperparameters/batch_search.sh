#!/bin/bash

if [[ "$#" -ne 2 ]]; then
    printf 'usage: bash batch_search.sh INITIALIZATION GPU_NUMBER\n' >&2
    exit 1
fi

initialization=$1
gpu=$2

for b in 512 4096 32768 262144; do
    output="/data/wcassidy/output/init_vs_batch/$initialization/batch_$b"
    mkdir -p "$output"
    cmd=$(cat <<EOF
cd ~/omnifoldTop
source ~/omnifoldTop/setup.sh
python ~/omnifoldTop/unfold.py \
  -d /data/wcassidy/ttbar/sim/mntuple_ttbar_[012]_parton_ejets_matched.npz \
  -s /data/wcassidy/ttbar/sim/mntuple_ttbar_6_parton_ejets_matched.npz \
  -g $gpu \
  -o $output \
  --batch-size $b \
  --nresamples 10 \
  -e bootstrap_model \
  --truth-known \
  --plot-correlations \
  --plot-history \
  -r gaussian_bump
EOF
       )
    singularity exec --nv --bind /data ~/omnifoldTop/baseml_tf_v0.1.33.sif bash <<EOF || $(error="$?"; printf "Failed in $b\n" >&2; exit "$error")
$cmd
EOF
    printf "$cmd\n" > "$output"/invocation.txt
done
