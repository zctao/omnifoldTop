#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M
#SBATCH --time=5:00:00
#SBATCH --output=%N-%j.out

cd $SLURM_TMPDIR

SOURCEDIR=~/omnifoldTop/topUnfolding
DATADIR=~/work/batch_output/TopNtupleAnalysis/ttbar/20200705/npz
OUTDIR=~/work/batch_output/OmniFold/20200712/test
DATA_INPUT=re_ttbar_19.npz
SIM_INPUT=rmu_ttbar_19.npz
OBS_TRAIN='th_pt th_eta th_phi th_m tl_pt tl_eta tl_phi tl_m'

module load python/3.7 cuda cudnn

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
cp $DATADIR/$DATA_INPUT $SLURM_TMPDIR/input/.
cp $DATADIR/$SIM_INPUT $SLURM_TMPDIR/input/.

# Start run
runcmd="python3 ${SOURCEDIR}/unfold.py -d ${SLURM_TMPDIR}/input/${DATA_INPUT} -s ${SLURM_TMPDIR}/input/${SIM_INPUT} -o ${SLURM_TMPDIR}/output -n -t -i 5 --observables-train ${OBS_TRAIN}"

echo "execute command: ${runcmd}"
$runcmd

# Output
cp $SLURM_TMPDIR/output/* $OUTDIR/.
