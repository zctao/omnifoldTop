#!/bin/bash

module load python/3.7 cuda cudnn

SOURCEDIR=~/omnifoldTop/topUnfolding
DATADIR=~/work/batch_output/TopNtupleAnalysis/ttbar/20200705/npz

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
# Install pre-downloaded packages
pip install ~/work/wheels/EnergyFlow-1.0.3-py2.py3-none-any.whl
pip install ~/work/wheels/sklearn-0.0.tar.gz
pip install ~/work/wheels/PyPDF2-1.26.0.tar.gz
# Install the rest of packages
pip install --no-index -r $SOURCEDIR/requirements_cedar_gpu.txt

# Prepare data
mkdir $SLURM_TMPDIR/input $SLURM_TMPDIR/output
# for now
#cp $SOURCEDIR/input/* $SLURM_TMPDIR/input/.
cp $DATADIR/*_ttbar.npz $SLURM_TMPDIR/input/.

# Start running
echo "python3 $SOURCEDIR/unfold.py -d ./input/re_ttbar.npz -s ./input/rmu_ttbar.npz -n -t"
