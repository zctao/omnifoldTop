#!/bin/bash
outdir=/home/ztao/atlasserv/OmniFoldOutputs/ToyData

for preset in 1d 2d_ind 2d_meas_cor 2d_true_cor 2d_cor; do
  fname_sim=${outdir}/toy_sim_${preset}_1e6.npz
  fname_obs=${outdir}/toy_obs_${preset}_1e6.npz

  python scripts/generate_toydata.py -n 1000000 -s ${fname_sim} -d ${fname_obs} -p ${preset}
done