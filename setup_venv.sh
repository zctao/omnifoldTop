#!/bin/bash
VENV_DIR=${1:-venv}
REQUIREMENTS=${2:-requirements_cedar.txt}

if [ -d "$VENV_DIR" ]; then
    echo "Activate virtual environment $VENV_DIR"
    source $VENV_DIR/bin/activate
else
    echo "Set up virtual environment $VENV_DIR"
    virtualenv --no-download $VENV_DIR
    source $VENV_DIR/bin/activate

    pip install --no-index --upgrade pip

    # Install packages
    # packaging, matplotlib==3.3.2, pandas==1.1.5, scipy, scikit_learn==0.23.2, tensorflow_gpu, # numpy
    pip install --no-index -r $REQUIREMENTS

    # needed by external.OmniFold.modplot
    pip install /home/ztao/work/wheels/PyPDF2-1.26.0.tar.gz 
fi
