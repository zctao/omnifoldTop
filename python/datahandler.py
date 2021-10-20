"""
Classes for working with OmniFold datasets.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from util import parse_input_name
from histogramming import calc_hist

def load_dataset(file_names, array_name='arr_0', allow_pickle=True, encoding='bytes', weight_columns=[]):
    """
    Load and return a structured numpy array from a list of npz files.

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of npz files to load. If columns in the file should be
        reweighted, provide the filename as "{path}*{reweight factor}".
        Columns identified in `weight_columns` will be multiplied by
        `reweight factor`.
    array_name : str, default: "arr_0"
        Name of the array to load from each file in `file_names`
    allow_pickle : bool, default: True
        Allow loading pickled object arrays. Loading pickled data can
        execute arbitrary code. If False, object arrays will fail to load.
    encoding : {"bytes", "ASCII", "latin1"}, default: "bytes"
        Text encoding to use when loading Python 2 strings in Python 3.
    weight_columns : str or sequence of str, default: []
        Names of columns to reweight, if reweighting is being used.

    Returns
    -------
    np.ndarray
        Arrays saved in each file, concatenated.

    Raises
    ------
    IOError
        If a file does not exist or can't be read.
    RuntimeError
        If a file doesn't contain an array named `array_name`.
    ValueError
        If a file contains an object array but `allow_pickle` is False.
    ValueError
        A file using reweighting doesn't contain `weight_columns`.

    See Also
    --------
    np.load : details of how NumPy loads files
    """
    data = None
    if not isinstance(file_names, list):
        file_names = [file_names]
    if not isinstance(weight_columns, list):
        weight_columns = [weight_columns]

    for fname in file_names:
        fn, rwfactor = parse_input_name(fname)

        npzfile = np.load(fn, allow_pickle=allow_pickle, encoding=encoding)
        di = npzfile[array_name]
        if len(di)==0:
            raise RuntimeError('There is no events in input file {}'.format(fname))

        # rescale total event weights for this input file
        if rwfactor != 1.:
            for wname in weight_columns:
                try:
                    di[wname] *= rwfactor
                except ValueError:
                    print('Unknown field name {}'.format(wname))
                    continue

        if data is None:
            data = di
        else:
            data  = np.concatenate([data, di])
        npzfile.close()

    return data

class DataHandler(Mapping):
    """
    Load data from a series of npy files.

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of npz files to load. If the weight column should be adjusted,
        provide the filename as "{path}*{reweight factor}".
    variable_names : list of str
        List of reco-level variables to read. If not provided, read all 
        variables in the dataset.
    variable_names_mc : list of str, optional
        List of truth-level variables to read. If not provided, skip reading
        truth-level variables
    weights_name : str, default: "w"
        Name of the event weight column.
    weights_name_mc : str, default: None
        Name of the mc weights
    vars_dict : dict, default: {}
        Use the key "vtype" to set the dtype of the loaded data. If not
        provided, defaults to "float".
    array_name : str, default: "arr_0"
        Name of the array to load from each file.

    Raises
    ------
    IOError
        If a file does not exist or can't be read.
    ValueError
        A file doesn't contain one of `variable_names` or `variable_names_mc`.
    RuntimeError
        A file doesn't contain an array named `array_name`.
    ValueError
        A file using reweighting doesn't contain `weights_name` or `weights_name_mc`.
    """

    def __init__(
        self,
        filepaths,
        variable_names,
        variable_names_mc=[],
        weights_name="w",
        weights_name_mc=None,
        vars_dict={},
        array_name="arr_0",
    ):
        # load data from npz files to numpy array
        tmpDataArr = load_dataset(filepaths, array_name=array_name, weight_columns=weights_name)
        #assert(tmpDataArr is not None)

        # reco level
        variable_names = _filter_variable_names(variable_names)

        # convert all fields to float
        dtypes = [(vname, vars_dict.get('vtype','float')) for vname in variable_names]
        self.data_reco = np.array(tmpDataArr[variable_names], dtype=dtypes)

        # event weight
        if weights_name:
            self.weights = tmpDataArr[weights_name].flatten()
        else:
            self.weights = np.ones(len(self.data_reco))

        # truth level
        if variable_names_mc:
            variable_names_mc = _filter_variable_names(variable_names_mc)

            # convert all fields to float
            dtypes_mc = [(vname, vars_dict.get('vtype','float')) for vname in variable_names_mc]
            self.data_truth = np.array(tmpDataArr[variable_names_mc], dtype=dtypes_mc)

            # mc weights
            if weights_name_mc:
                self.weights_mc = tmpDataArr[weights_name_mc].flatten()
            else:
                self.weights_mc = np.ones(len(self.data_truth))
        else:
            self.data_truth = None
            self.weights_mc = None

        # for now
        self.pass_reco = np.full(len(self.data_reco), True)
        if self.data_truth is not None:
            self.pass_truth = np.full(len(self.data_truth), True)
        else:
            self.pass_truth = None

    def __len__(self):
        """
        Get the number of events in the dataset.

        Returns
        -------
        non-negative int
        """
        return len(self.data_reco)

    def __contains__(self, variable):
        """
        Check if a variable is in the dataset.

        Parameters
        ----------
        variable : str

        Returns
        -------
        bool
        """
        inReco = variable in self.data_reco.dtype.names

        if self.data_truth is None:
            inTruth = False
        else:
            inTruth = variable in self.data_truth.dtype.names

        return inReco or inTruth

    def _get_array(self, variable):
        """
        Return a 1D numpy array of the variable

        Parameters
        ----------
        variable : str
            Name of the variable

        Returns
        -------
        np.ndarray of shape (n_events,)
        """
        if variable in self.data_reco.dtype.names:
            return self.data_reco[str(variable)]

        if self.data_truth is not None:
            if variable in self.data_truth.dtype.names:
                return self.data_truth[str(variable)]

        # no 'variable' in the data arrays
        return None

    def __getitem__(self, features):
        """
        Retrieve features from each event in the dataset.

        Returns a view (NOT copy) of self.data if possible. Otherwise,
        tries to make a new array from self.data.

        Parameters
        ----------
        features : array-like of str
            Names of the features to extract from each event. The shape of
            the returned array will reflect the shape of this array.

        Returns
        -------
        np.ndarray of shape (n_events, *features.shape)

        Raises
        ------
        KeyError
            If a variable name in `features` is not in the dataset.
        """
        ndim_features = np.asarray(features).ndim
        if ndim_features == 0:
            if features in self:
                 # Can't index data by np Unicode arrays, have to
                 # convert back to str first.
                #return self.data[str(features)]
                return self._get_array(features)
            # special cases
            elif '_px' in features:
                var_pt = features.replace('_px', '_pt')
                var_phi = features.replace('_px', '_phi')
                arr_pt = self[var_pt]
                arr_phi = self[var_phi]
                return arr_pt * np.cos(arr_phi)
            elif '_py' in features:
                var_pt = features.replace('_py', '_pt')
                var_phi = features.replace('_py', '_phi')
                arr_pt = self[var_pt]
                arr_phi = self[var_phi]
                return arr_pt * np.sin(arr_phi)
            elif '_pz' in features:
                var_pt = features.replace('_pz', '_pt')
                var_eta = features.replace('_pz', '_eta')
                arr_pt = self[var_pt]
                arr_eta = self[var_eta]
                return arr_pt * np.sinh(arr_eta)
            else:
                raise KeyError(
                    "Unknown variable {}. \nAvailable variable names: {}".format(
                        features,
                        list(self.keys()),
                    )
                )
        else:
            # ndarray of shape (n_events, <feature shape>)
            X = np.stack([self[varnames] for varnames in features], axis=1)
            return X

    def __iter__(self):
        """
        Create an iterator over the variable names in the dataset.

        Returns
        -------
        iterator of strings
        """
        if self.data_truth is None:
            return iter(self.data_reco.dtype.names)
        else:
            return iter(
                list(self.data_reco.dtype.names) +
                list(self.data_truth.dtype.names)
                )

    def sum_weights(self, reco_level=True):
        """
        Get sum of event weights

        Parameters
        ----------
        reco_level: bool, default: True

        Return
        ------
        If reco_level is True, return sum of self.weights, otherwise return sum
        of self.weights_mc
        """
        if reco_level:
            return self.weights.sum()
        else:
            return self.weights_mc.sum()

    def rescale_weights(
        self,
        factors=1.,
        reweighter=None,
    ):
        """
        Rescale event weights of the dataset

        Parameters
        ----------
        factors : float or np.ndarray of the same dimension as self.weights
            Factors to rescale the event weights
        reweighter : reweight.Reweighter, optional
            A function that takes events and returns event weights, and the
            variables it expects.

        Notes
        -----
        Order of operations: reweighting, rescaling
        """
        # reweight sample
        if reweighter is not None:
            self.weights *= reweighter.func(self[reweighter.variables])
            self.weights_mc *= reweighter.func(self[reweighter.variables])

        # rescale
        self.weights *= factors
        self.weights_mc *= factors

    def get_weights(
        self,
        bootstrap=False,
        reco_level=True
    ):
        """
         Get event weights for the dataset.

        Parameters
        ----------
        bootstrap : bool, default: False
            Multiply each weight by a random value drawn from a Poisson
            distribution with lambda = 1.
        reco_level : bool, default: True
            If True, return reco-level event weights ie. self.weights
            Otherwise, return MC truth weights self.weights_mc

        Returns
        -------
        np.ndarray of numbers, shape (nevents,)
        """
        if reco_level:
            weights = self.weights.copy()
        else:
            weights = self.weights_mc.copy()

        # bootstrap
        if bootstrap:
            weights *= np.random.poisson(1, size=len(weights))

        return weights

    def get_correlations(self, variables):
        """
        Calculate the correlation matrix between several variables.

        Parameters
        ----------
        variables : sequence of str
            Names of the variables to include in the correlation matrix.

        Returns
        -------
        pandas.DataFrame
        """
        df = pd.DataFrame({var: self[var] for var in variables}, columns=variables)
        correlations = df.corr()
        return correlations

    def get_histogram(self, variable, bin_edges, weights=None, density=False):
        """
        Retrieve the histogram of a weighted variable in the dataset.

        Parameters
        ----------
        variable : str
            Name of the variable in the dataset to histogram.
        weights : array-like of shape (nevents,) or (nweights, nevents) or None
            Array of per-event weights. If 2D, then a sequence of different
            per-event weightings. If None, use self.weights or self.weights_mc
        bin_edges : array-like of shape (nbins + 1,)
            Locations of the edges of the bins of the histogram.

        Returns
        -------
        A Hist object if `weights` is 1D, a list of Hist objects if `weights` 
            is 2D.
        """
        if weights is None:
            if variable in self.data_truth.dtype.names: # mc truth level
                weights = self.weights_mc
            else: # reco level
                weights = self.weights

        if isinstance(weights, np.ndarray):
            if weights.ndim == 1: # if weights is a 1D array
                varr = self[variable]
                # check the weight array length is the same as the variable array
                assert(len(varr) == len(weights))
                return calc_hist(varr, weights=weights, bins=bin_edges, density=density)
            elif weights.ndim == 2: # make the 2D array into a list of 1D array
                return self.get_histogram(variable, bin_edges, list(weights), density)
            else:
                raise RuntimeError("Only 1D or 2D array or a list of 1D array of weights can be processed.")
        elif isinstance(weights, list): # if weights is a list of 1D array
            hists = []
            for w in weights:
                h = self.get_histogram(variable, bin_edges, w, density)
                hists.append(h)
            return hists
        else:
            raise RuntimeError("Unknown type of weights: {}".format(type(weights)))

def _filter_variable_names(variable_names):
    """
    Normalize a list of variables.

    Replaces Cartesian variables with equivalent cylindrical variables
    and removes duplicate variable names.

    Parameters
    ----------
    variable_names : iterable of str
        Variable names to process. If a variable ends in ``_px``,
        ``_py``, or ``_pz``, it is interpreted as a Cartesian variable.

    Returns
    -------
    list of str
        Processed variable names. Not guaranteed to preserve order from
        the input iterable.
    """
    varnames_skimmed = set()

    for vname in variable_names:
        if '_px' in vname:
            vname_pt = vname.replace('_px', '_pt')
            vname_phi = vname.replace('_px', '_phi')
            varnames_skimmed.add(vname_pt)
            varnames_skimmed.add(vname_phi)
        elif '_py' in vname:
            vname_pt = vname.replace('_py', '_pt')
            vname_phi = vname.replace('_py', '_phi')
            varnames_skimmed.add(vname_pt)
            varnames_skimmed.add(vname_phi)
        elif '_pz' in vname:
            vname_pt = vname.replace('_pz', '_pt')
            vname_eta = vname.replace('_pz', '_eta')
            varnames_skimmed.add(vname_pt)
            varnames_skimmed.add(vname_eta)
        else:
            varnames_skimmed.add(vname)

    return list(varnames_skimmed)

def standardize_dataset(features):
        """
        Standardize the distribution of a set of features.

        Adjust the dataset so that the mean is 0 and standard deviation is 1.

        Parameters
        ----------
        features : array-like (n_events, *feature_shape)
            Array of data. The data is interpreted as a series of feature
            arrays, one per event. Standardization is performed along the
            event axis.

        Returns
        -------
        np.ndarray of shape (n_events, *feature_shape)
            Standardized dataset.

        Examples
        --------
        >>> a = np.asarray([
        ...     [1, 2, 3],
        ...     [4, 5, 6],
        ... ])
        >>> datahandler.standardize_dataset(a)
        array([[-1., -1., -1.],
               [ 1.,  1.,  1.]])
        """
        centred_at_zero = features - np.mean(features, axis=0)
        deviation_one = centred_at_zero / np.std(centred_at_zero, axis=0)

        return deviation_one

# Toy data
class DataToy(DataHandler):
    """
    A randomly generated toy dataset.

    The truth distribution is sampled from a normal distribution specified
    by `mu` and `sigma`. The reconstructed distribution is obtained by adding
    Gaussian noise with standard deviation 1 to the truth distribution.

    Parameters
    ----------
    nevents : positive int
        Number of events in the dataset
    mu : float, default: 0
        Mean of the truth distribution.
    sigma : positive float, default: 1
        Standard deviation of the truth distribution.
    """
    def __init__(self, nevents, mu=0., sigma=1.):
        # generate toy data
        # truth level
        var_truth = np.random.normal(mu, sigma,nevents)

        # detector smearing
        epsilon = 1. # smearing width
        var_reco = np.array([(x + np.random.normal(0, epsilon)) for x in var_truth])

        self.data_reco = np.array(var_reco, dtype=[('x_reco','float')])
        self.data_truth = np.array(var_truth, dtype=[('x_truth', 'float')])

        #self.data = np.array([(x,y) for x,y in zip(var_reco, var_truth)],
        #                     dtype=[('x_reco','float'), ('x_truth','float')])

        self.weights = np.ones(nevents)
        self.weights_mc = np.ones(nevents)

        self.pass_reco = np.full(nevents, True)
        self.pass_truth = np.full(nevents, True)
