#!/bin/bash
VENV_DIR=${1:-venv}

export SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# cf. https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself

source $SOURCE_DIR/scripts/setup_venv.sh $VENV_DIR $SOURCE_DIR/requirements.txt

export PYTHONPATH=$SOURCE_DIR/python:$SOURCE_DIR:$PYTHONPATH
