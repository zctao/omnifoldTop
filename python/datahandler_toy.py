import numpy as np
from numpy.random import default_rng
rng = default_rng()

from datahandler import DataHandlerBase

# Toy data
class DataHandlerToy(DataHandlerBase):
    """
    A randomly generated toy dataset.

    The truth distribution is sampled from a normal distribution specified
    by `mu` and `sigma`. The reconstructed distribution is obtained by adding
    Gaussian noise with standard deviation 1 to the truth distribution.
    """

    def __init__(self):
        super().__init__()

    def generate(
        self,
        nevents,
        varnames = ['x'],
        mean = 0.,
        covariance = 1.,
        covariance_meas = 1.,
        eff = 1.,
        acc = 1.,
        dummy_value = -10.
        ):
        """
        Parameters
        ----------
        nevents : positive int
            Number of events in the dataset
        varname :  list of str, default: ['x']
            List of variable names
        mean : float or sequence of float, default: 0
            Mean of the truth distribution.
        covariance : float or 1D array of float or 2D array of float, default: 1
            Covariance for generating toy data
        covariance_meas : float or 1D array of float or 2D array of float, default: 1
            Covariance for detector smearing
        eff : positive float, default: 1
            Reconstruction efficiency. Fraction of events in truth that are also in reco
        acc : positivee float, default: 1
            Acceptance. Fraction of the events in reco that also in truth
        dummy_value : float, default -10.
            Value assigned to events that are in reco but not truth or in truth but not reco
        """

        if np.asarray(mean).ndim == 0: # a float
            mean = np.asarray([mean]) # change to a 1D array
        else:
            mean = np.asarray(mean)

        if np.asarray(covariance).ndim == 0:
            cov_t = np.diag([covariance]*len(varnames))
        elif np.asarray(covariance).ndim == 1:
            cov_t = np.diag(covariance)
        elif np.asarray(covariance).ndim == 2:
            cov_t = covariance
        else:
            print(f"ERROR: covariance_true has to be a float, 1D array of float, or 2D array of float")
            return

        if np.asarray(covariance_meas).ndim == 0:
            cov_m = np.diag([covariance_meas]*len(varnames))
        elif np.asarray(covariance_meas).ndim == 1:
            cov_m = np.diag(covariance_meas)
        elif np.asarray(covariance_meas).ndim == 2:
            cov_m = covariance_meas
        else:
            print(f"ERROR: covariance_meas has to be a float, 1D array of float, or 2D array of float")
            return

        # generate toy data
        # truth level
        dtype_truth = [(v+'_truth', 'float') for v in varnames]
        arr_truth = rng.multivariate_normal(mean, cov=cov_t, size=nevents)
        self.data_truth = np.rec.fromarrays(arr_truth.T, dtype=dtype_truth)

        # detector level
        # define detector smearing
        def measure(*data):
            d = np.asarray(data)
            s = rng.multivariate_normal([0.]*len(data), cov=cov_m)
            return tuple(d+s)

        #after smearing
        dtype_reco = [(v+'_reco', 'float') for v in varnames]
        self.data_reco = np.array([measure(*data) for data in self.data_truth], dtype=dtype_reco)

        # efficiency
        if eff < 1:
            self.pass_reco = rng.binomial(1, eff, nevents).astype(bool)
            self.data_reco[~self.pass_reco] = dummy_value
        else:
            self.pass_reco = np.full(nevents, True)

        # acceptance
        if acc < 1:
            self.pass_truth = rng.binomial(1, acc, nevents).astype(bool)
            self.data_truth[~self.pass_truth] = dummy_value
        else:
            self.pass_truth = np.full(nevents, True)

        # all event weights are one for now
        self.weights = np.ones(nevents)
        self.weights_mc = np.ones(nevents)

    def save_data(self, filepath, save_weights=True, save_pass=True):
        d = {'reco' :  self.data_reco, 'truth' : self.data_truth}

        if save_weights:
            d['weights'] = self.weights
            d['weights_mc'] = self.weights_mc

        if save_pass:
            d['pass_reco'] = self.pass_reco
            d['pass_truth'] = self.pass_truth

        np.savez(filepath, **d)

    def load_data(self, filepaths):
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        for fpath in filepaths:
            with np.load(fpath) as f:
                if self.data_reco is None:
                    self.data_reco = f['reco']
                else:
                    self.data_reco = np.concatenate((self.data_reco, f['reco']))

                if self.data_truth is None:
                    self.data_truth = f['truth']
                else:
                    self.data_truth = np.concatenate((self.data_truth, f['truth']))

                wtmp = f['weights'] if 'weights' in f else np.ones(len(self.data_reco))
                if self.weights is None:
                    self.weights = wtmp
                else:
                    self.weights = np.concatenate((self.weights, wtmp))

                wmctmp = f['weights_mc'] if 'weights_mc' in f else np.ones(len(self.data_truth))
                if self.weights_mc is None:
                    self.weights_mc = wmctmp
                else:
                    self.weights_mc = np.concatenate((self.weights_mc, wmctmp))

                preco = f['pass_reco'] if 'pass_reco' in f else np.full(len(self.data_reco), True)
                if self.pass_reco is None:
                    self.pass_reco = preco
                else:
                    self.pass_reco = np.concatenate((self.pass_reco, preco))

                ptruth = f['pass_truth'] if 'pass_truth' in f else np.full(len(self.data_truth), True)
                if self.pass_truth is None:
                    self.pass_truth = ptruth
                else:
                    self.pass_truth = np.concatenate((self.pass_truth, ptruth))