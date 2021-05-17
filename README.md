# omnifoldTop

Omnifold for ttbar l+jets final states with potential to deal with non-negligible backgrounds.

See https://github.com/ericmetodiev/OmniFold.git for the original repo from the OmniFold authors.

## Install

    git clone https://github.com/zctao/omnifoldTop.git
    
The current version depends on some plotting utilities from the original OmniFold package. 
To get the original OmniFold packakge:

    git clone https://github.com/ericmetodiev/OmniFold.git omnifoldTop/external/OmniFold

### Requirements (TODO)

    packaging, matplotlib==3.3.2, pandas==1.1.5, numpy, scipy, scikit_learn==0.23.2, tensorflow
    
Needed by the original OmniFold package:

    PyPDF2
    
It is also possible to run the application in a container. 
A compatible container can be obtained from `docker://triumfmlutils/baseml_tf`:

    singularity pull docker://triumfmlutils/baseml_tf:v0.1.33
    singularity run --nv [--bind DATADIR] baseml_tf_v0.1.33.sif

## Run

Set up environment and Python path:

    cd omnifoldTop
    source setup.sh
    
The main application to run is

    python3 unfold.py -d DATA_FILES -s SIMULATION_FILES \
                      [-o OUTPUT_DIRECTORY] \
                      [-i NUMBER_OF_ITERATIONS] \
                      [--observables-train LIST_OF_VARRIABLES_USED_IN_TRAININGS] \
                      [--observables LIST_OF_VARIABLES_TO_UNFOLD] \
                      
                      
To see all argument options:

    python3 unfold.py -h

Input files are expected to be numpy structured arrays. In case the DATA_FILES are actually pseudo data from MC simulations, a flag `-t` can be added to indicate the "MC truth" is known and can be used to evaluate the performance.

In case one wishes to reuse the previously trained results, an option `--unfolded-weights PATH_TO_UNFOLDED_WEIGHTS_FILES` can be used to read the event weights from the specified files and apply them directly to other variables.

### Other helper scripts

- `evaluateModels.py`: if input data files are pseudo data, i.e. MC truth is available, `evaluateModels.py` can be used to reweight simulation truth directly to the pseudo data truth and compare the reweighted distribution with the actual MC truth in the pseudo data. The goal is to evaluate the performance of the classifier model without running OmniFold iterations.

        python3 evaluateModels.py -d DATA_FILES -s SIMULATION_FILES \
                                  [-o OUTPUT_DIRECTORY] \
                                  [--observables LIST_OF_VARIABLES_TO_UNFOLD]

    To see all argument options:

        python3 evaluateModels.py -h

- `scripts/rootReader.py`: convert root files into npz files containing numpy structure arrays.

        python3 scripts/rootReader.py ROOT_FILES \
                                      --truth-level parton|particle \
                                      [-o OUTPUT_NAME]

- `scripts/makeRunScript.py`: generate a bash script to run multiple unfolding iterations with different parameters given a run configuration file.

        python3 scripts/makeRunScript.py RUNCONFIG [-o OUTPUT_NAME]

    An example of the run config file is provided in `configs/run/basic_tests.json`.
