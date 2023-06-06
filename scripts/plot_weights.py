import os
import h5py

import plotter

def plot_weights(
    fpath_weights,
    outputdir='.',
    array_name="unfolded_weights",
    logy=False
    ):
    with h5py.File(fpath_weights, 'r') as f:
        weights = f[array_name]
        # expected shape: (nruns, niterations, nevents)
        assert(weights.ndim == 3)

        # all weights final
        plotter.plot_LR_distr(
            os.path.join(outputdir, "weights_distribution"),
            [weights[:,-1,:].ravel()],
            xlabel = "weight",
            logy=logy
        )

        # weights from each run
        nruns = weights.shape[0]
        plotter.plot_LR_distr(
            os.path.join(outputdir, "weights_distribution_allruns"),
            [weights[r,-1,:] for r in range(nruns)],
            labels = [f"run-{r}" for r in range(nruns)],
            xlabel = "weight",
            logy=logy
        )

        # weights from each iteration
        niterations = weights.shape[1]
        plotter.plot_LR_distr(
            os.path.join(outputdir, "weights_distribution_alliters"),
            [weights[:,i,:].ravel() for i in range(niterations)],
            labels = [f"iter-{i+1}" for i in range(niterations)],
            xlabel = "weight",
            logy=logy
        )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("fpath_weights", type=str, help="File path to weights h5 file")
    parser.add_argument("-o", "--outputdir", type=str, default=".", help="Output directory")
    parser.add_argument("--array-name", type=str, default="unfolded_weights",
                        help="Name of the weight array")
    parser.add_argument("--logy", action="store_true", 
                        help="If true, use log scale for y-axis")
    
    args = parser.parse_args()

    plot_weights(**vars(args))