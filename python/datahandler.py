"""
Classes for working with OmniFold datasets.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from util import parse_input_name, normalize_histogram
# for now
import external.OmniFold.modplot as modplot

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
    wname : str, default: "w"
        Name of the event weight column.
    variable_names : list of str, optional
        List of variables to read. If not provided, read all variables in
        the dataset.
    vars_dict : dict, default: {}
        Use the key "vtype" to set the dtype of the loaded data. If not
        provided, defaults to "float".
    array_name : str, default: "arr_0"
        Name of the array to load from each file.

    Raises
    ------
    IOError
        If a file does not exist or can't be read.
    RuntimeError
        A file doesn't contain one of `variable_names`.
    RuntimeError
        A file doesn't contain an array named `array_name`.
    ValueError
        A file using reweighting doesn't contain `wname`.
    """

    def __init__(
        self,
        filepaths,
        wname="w",
        variable_names=None,
        vars_dict={},
        array_name="arr_0",
    ):
        self.weight_name = wname # name of event weights

        # load data from npz files to numpy array
        tmpDataArr = load_dataset(filepaths, array_name=array_name, weight_columns=wname)
        assert(tmpDataArr is not None)

        if not variable_names:
            # if no variable name list is provided, read everything
            variable_names = list(tmpDataArr.dtype.names)
        else:
            # add wname to the list
            if wname and not wname in variable_names:
                variable_names.append(wname)

            variable_names = _filter_variable_names(variable_names)

            # check all variable names are available
            for vname in variable_names:
                if not vname in tmpDataArr.dtype.names:
                    raise RuntimeError("Unknown variable name {}".format(vname))

        # convert all fields to float
        dtypes = [(vname, vars_dict.get('vtype','float')) for vname in variable_names]
        self.data = np.array(tmpDataArr[variable_names], dtype=dtypes)

        # sum of event weights
        self.sumw = len(self.data)
        if wname and wname in self.data.dtype.names:
            self.sumw = self.data[wname].sum()

    def __len__(self):
        """
        Get the number of events in the dataset.

        Returns
        -------
        non-negative int
        """
        return len(self.data)

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
        return variable in self.data.dtype.names

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
                return self.data[features]
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
        return iter(self.data.dtype.names)

    def get_weights(
            self,
            unweighted=False,
            bootstrap=False,
            normalize=False,
            rw_type=None,
            vars_dict={},
    ):
        """
        Get event weights for the dataset.

        Parameters
        ----------
        unweighted : bool, default: False
            Ignore weights saved in the dataset and use unity instead.
        bootstrap : bool, default: False
            Multiply each weight by a random value drawn from a Poisson
            distribution with lambda = 1.
        normalize : bool, default: False
            If True, normalize the weights by expressing them as the ratio
            to the mean of the weights.
        rw_type : str, optional
            Multiply the weights by a predefined reweighting function.
        vars_dict : dict, default: {}
            Dict describing how to access variables.

        Returns
        -------
        np.ndarray of numbers, shape (nevents,)

        See Also
        --------
        datahandler.DataHandler._reweight_sample

        Notes
        -----
        Normalization is applied after reweighting and before
        bootstrapping.
        """
        if unweighted or not self.weight_name:
            weights = np.ones(len(self))
        else:
            # always return a copy of the original weight array in self.data
            weights = self[self.weight_name].copy()
            assert(weights.base is None)

        # reweight sample if needed
        if rw_type is not None:
            weights *= self._reweight_sample(rw_type, vars_dict)

        # normalize to len(self.data)
        if normalize:
            weights /= np.mean(weights)

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

    def get_histogram(self, variable, weights, bin_edges, normalize=False):
        """
        Retrieve the histogram of a weighted variable in the dataset.

        Parameters
        ----------
        variable : str
            Name of the variable in the dataset to histogram.
        weights : array-like of shape (nevents,) or (nweights, nevents)
            Array of per-event weights. If 2D, then a sequence of different
            per-event weightings.
        bin_edges : array-like of shape (nbins + 1,)
            Locations of the edges of the bins of the histogram.

        Returns
        -------
        histogram : np.ndarray of shape (nbins,) or (nweights, nbins)
            Bin heights for `variable` with events weighted according to
            `weights`. If `weights` is 2D, then returns a sequence of
            histograms, one per set of weights.
        errors : np.ndarray of shape (nbins,) or (nweights, nbins)
            Uncertainties on the bin heights. If `weights` is 2D, then
            returns a sequence of uncertainties, one per set of weights.
        """
        if isinstance(weights, np.ndarray):
            if weights.ndim == 1: # if weights is a 1D array
                varr = self[variable]
                # check the weight array length is the same as the variable array
                assert(len(varr) == len(weights))
                hist, hist_err = modplot.calc_hist(varr, weights=weights, bins=bin_edges, density=False)[:2]
                if normalize:
                    normalize_histogram(bin_edges, hist, hist_err)
                return hist, hist_err
            elif weights.ndim == 2: # make the 2D array into a list of 1D array
                return self.get_histogram(variable, list(weights), bin_edges, normalize)
            else:
                raise RuntimeError("Only 1D or 2D array or a list of 1D array of weights can be processed.")
        elif isinstance(weights, list): # if weights is a list of 1D array
            hists, hists_err = [], []
            for w in weights:
                h, herr = self.get_histogram(variable, w, bin_edges, normalize)
                hists.append(h)
                hists_err.append(herr)
            return np.asarray(hists), np.asarray(hists_err)
        else:
            raise RuntimeError("Unknown type of weights: {}".format(type(weights)))

    def _reweight_sample(self, rw_type, vars_dict):
        """
        Return sample weights calculated by a function of the dataset.

        Parameters
        ----------
        rw_type : {Falsey, "linear_th_pt", "gaussian_bump", "gaussian_tail"}
            Name of the reweighting function to use. See Notes for the
            function corresponding to each value.
        vars_dict : dict
            Variable configuration dict mapping variables to their name in
            the dataset.

        Returns
        -------
        np.ndarray of shape (nevents,)
             Event weights.

        Raises
        ------
        RuntimeError
            If an unknown `rw_type` is passed.

        Notes
        -----
        =================== ===============================================
        Value of `rw_type`  Reweighting function
        =================== ===============================================
        Any Falsey value    ``1``
        ``"linear_th_pt"``  ``1 + th_pt / 800``
        ``"gaussian_bump"`` ``1 + 0.5 * exp( -((mtt - 800) / 100) ** 2)``
        ``"gaussian_tail"`` ``1 + 0.5 * exp( -((mtt - 2000) / 1000) ** 2)``
        =================== ===============================================
        """
        if not rw_type:
            return 1.
        elif rw_type == 'linear_th_pt':
            # truth-level hadronic top pt
            assert('th_pt' in vars_dict)
            varname_thpt = vars_dict['th_pt']['branch_mc']
            th_pt = self[varname_thpt]
            # reweight factor
            rw = 1. + 1/800.*th_pt
            return rw
        elif rw_type == 'gaussian_bump':
            # truth-level ttbar mass
            assert('mtt' in vars_dict)
            varname_mtt = vars_dict['mtt']['branch_mc']
            mtt = self[varname_mtt]
            #reweight factor
            k = 0.5
            m0 = 800.
            sigma = 100.
            rw = 1. + k*np.exp( -( (mtt-m0)/sigma )**2 )
            return rw
        elif rw_type == 'gaussian_tail':
            assert('mtt' in vars_dict)
            varname_mtt = vars_dict['mtt']['branch_mc']
            mtt = self[varname_mtt]
            #reweight factor
            k = 0.5
            m0 = 2000.
            sigma = 1000.
            rw = 1. + k*np.exp( -( (mtt-m0)/sigma )**2 )
            return rw
        else:
            raise RuntimeError("Unknown reweighting type: {}".format(rw_type))

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
        self.weight_name = ''

        # generate toy data
        # truth level
        var_truth = np.random.normal(mu, sigma,nevents)

        # detector smearing
        epsilon = 1. # smearing width
        var_reco = np.array([(x + np.random.normal(0, epsilon)) for x in var_truth])

        self.data = np.array([(x,y) for x,y in zip(var_reco, var_truth)],
                             dtype=[('x_reco','float'), ('x_truth','float')])
