#!/bin/bash
export DATA_DIR=${1:-$HOME/data}

export SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# cf. https://stackoverflow.com/questions/59895/how-to-get-the-source-directory-of-a-bash-script-from-within-the-script-itself

export PYTHONPATH=${SOURCE_DIR}/python:${SOURCE_DIR}/scripts:${SOURCE_DIR}:$PYTHONPATH