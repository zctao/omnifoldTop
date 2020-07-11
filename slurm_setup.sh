#!/bin/bash

module load python/3.7 cuda cudnn

SOURCEDIR=~/omnifoldTop/topUnfolding

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
cp $SOURCEDIR/input/* $SLURM_TMPDIR/input/.

# Start running
