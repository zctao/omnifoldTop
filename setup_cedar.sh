#!/bin/bash
VENV_DIR=${1:-venv}

export SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# cf. https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself

module load python/3.7 cuda cudnn

# set up virtual environment
source $SOURCE_DIR/setup_venv.sh $VENV_DIR $SOURCE_DIR/requirements_cedar.txt

export PYTHONPATH=$SOURCE_DIR/python:$SOURCE_DIR:$PYTHONPATH
