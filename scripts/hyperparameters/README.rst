====================================
 Hyperparameter exploration scripts
====================================

Contains several scripts to explore hyperparameters of the OmniFold
neural networks. Running the searches is generally done with one of
the bash scripts; analyzing the results is generally done with one of
the Python scripts.

A script exploring hyperparameters `p1` and `p2` will generally output
to `/<common_root>/<p1 value>/<p2 value>/`. The data loading scripts
in `util.py` use that format and produce a Pandas `DataFrame`
indexed by columns including `p1` and `p2`.

Analysis scripts should save figures to the `figs/` subdirectory,
which is excluded from tracking in git. Python scripts should be
formatted by `psf/black <https://github.com/psf/black>`_.

Fixed parameters
================
- Input dataset: 600k each of data and sim => 1.2M events
- Network: 3 layers, 100 nodes each
- Gaussian reweighting stress test

Initialization vs batch size
============================
- Explore effect of initalization scheme and batch size on unfolding quality
- Explore potential interaction between initialization and batch size
- Set batch size by `-b` command line argument
- Set kernel initializer manually in model.py
- Run using batch_search.sh

Phase space
-----------
- Batch size: :math:`2^9 = 512`, :math:`2^{12} = 4096`, :math:`2^{15} = 32768`, :math:`2^{18} = 262144`
- Initialization: Glorot and He, normal and uniform.

Phase space: 16 elements.

Outputs
-------
- Training time: bi_time.py, bi_time.csv, figs/timing.png
- χ^2/NDF: mean_chisq.py, chisq.py, figs/chisq_batch_individual_vars.png, figs/chisq_vs_batch*.png
- Bin errors: bin_errors.py, figs/bin_errors.png

Learning rate
=============
- Look at the Adam learning rate
- Default value: 0.001
- Values to try: 1e0 to 1e-5
- Set using OF_LR argument var (uncomitted code modification in model.py to read it)
- For high learning rates generating histograms sometimes raised ValueError; just catch and move on

Outputs
-------
- LR 1, 0.1: end up with infinite loss
- loss_history.py, lr_chisq.py, lr_spread.py, lr_time.csv, lr_variables.py
