#!/bin/bash
VENV_DIR=${1:-venv}
REQUIREMENTS=${2:-requirements.txt}

if [ -d "$VENV_DIR" ]; then
    echo "Activate virtual environment $VENV_DIR"
    source $VENV_DIR/bin/activate
else
    echo "Set up virtual environment $VENV_DIR"
    virtualenv --no-download $VENV_DIR
    source $VENV_DIR/bin/activate

    pip install --no-index --upgrade pip

    # Install packages
    pip install --no-index -r $REQUIREMENTS
fi
