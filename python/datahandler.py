"""
Classes for working with OmniFold datasets.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import util
from histogramming import calc_hist, calc_hist2d
import FlattenedHistogram as fh

from numpy.random import default_rng
rng = default_rng()

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
        fn, rwfactor = util.parse_input_name(fname)

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

        # overflow/underflow flags to be set later
        self.underflow_overflow_reco = False
        self.underflow_overflow_truth = False

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
        return self._in_data_reco(variable) or self._in_data_truth(variable)

    def __getitem__(self, features):
        """
        Retrieve features from the dataset.

        Return arrays containing only valid events. It is equivalent to 
        self.get_arrays(features, valid_only=True)

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
        return self.get_arrays(features, valid_only=True)

    def _in_data_reco(self, variable):
        if self.data_reco is None:
            return False
        else:
            return variable in self.data_reco.dtype.names

    def _in_data_truth(self, variable):
        if self.data_truth is None:
            return False
        else:
            return variable in self.data_truth.dtype.names

    def get_arrays(self, features, valid_only=False):
        """
        Retrieve features from each event in the dataset.

        Returns an array of features from self.data_reco or self.data_truth. 

        Parameters
        ----------
        features : array-like of str
            Names of the features to extract from each event. The shape of
            the returned array will reflect the shape of this array.
        valid_only : bool, default False
            If True, include only valid events (pass_reco and/or pass_truth),
            otherwise include all events.

        Returns
        -------
        np.ndarray of shape (n_events, *features.shape)

        Raises
        ------
        KeyError
            If a variable name in `features` is not in the dataset.
        """

        ndim_features = np.asarray(features).ndim

        if valid_only:
            # array filters for valid events
            sel = True
            varlist = [features] if ndim_features == 0 else list(features)
            for v in varlist:
                if self._in_data_reco(v): # reco level
                    sel &= self.pass_reco
                elif self._in_data_truth(v): 
                    sel &= self.pass_truth
                else:
                    raise KeyError(f"Unknown variable {v}")

            return self.get_arrays(features, valid_only=False)[sel]

        # not valid only
        if ndim_features == 0:
            if self._in_data_reco(features): # reco level
                # Can't index data by np Unicode arrays, have to
                # convert back to str first.
                return self.data_reco[str(features)]
            elif self._in_data_truth(features): # truth level
                return self.data_truth[str(features)]

            # special cases
            elif '_px' in features:
                var_pt = features.replace('_px', '_pt')
                var_phi = features.replace('_px', '_phi')
                arr_pt = self.get_arrays(var_pt)
                arr_phi = self.get_arrays(var_phi)
                return arr_pt * np.cos(arr_phi)
            elif '_py' in features:
                var_pt = features.replace('_py', '_pt')
                var_phi = features.replace('_py', '_phi')
                arr_pt = self.get_arrays(var_pt)
                arr_phi = self.get_arrays(var_phi)
                return arr_pt * np.sin(arr_phi)
            elif '_pz' in features:
                var_pt = features.replace('_pz', '_pt')
                var_eta = features.replace('_pz', '_eta')
                arr_pt = self.get_arrays(var_pt)
                arr_eta = self.get_arrays(var_eta)
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
            X = np.stack([self.get_arrays(varnames) for varnames in features], axis=1)
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
            return self.weights[self.pass_reco].sum()
        else:
            return self.weights_mc[self.pass_truth].sum()

    def rescale_weights(
        self,
        factors=1.,
        reweighter=None,
    ):
        """
        Rescale event weights of the dataset

        Parameters
        ----------
        factors : float
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
            # reweight events that pass both reco and truth level cuts
            sel = self.pass_reco & self.pass_truth
            varr = self.get_arrays(reweighter.variables, valid_only=False)[sel]
            self.weights[sel] *= reweighter.func(varr)
            if self.weights_mc is not None:
                self.weights_mc[sel] *= reweighter.func(varr)

        # rescale
        self.weights[self.pass_reco] *= factors
        if self.weights_mc is not None:
            self.weights_mc[self.pass_truth] *= factors

    def get_weights(
        self,
        bootstrap=False,
        reco_level=True,
        valid_only=True
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
        valid_only : bool, default: True
            If True, return weights of valid events only ie. pass_reco or 
            pass_truth, otherwise return all event weights including dummy ones

        Returns
        -------
        np.ndarray of numbers, shape (nevents,)
        """
        if reco_level:
            weights = self.weights.copy()
            sel = self.pass_reco
        else:
            weights = self.weights_mc.copy()
            sel = self.pass_truth

        if valid_only:
            weights = weights[sel]

        # bootstrap
        if bootstrap:
            weights *= rng.poisson(1, size=len(weights))

        return weights

    def get_correlations(self, variables, weights=None):
        """
        Calculate the correlation matrix between several variables.

        Parameters
        ----------
        variables : sequence of str
            Names of the variables to include in the correlation matrix.
        weights : array-like of shape (nevents,), default None
            Event weigts for computing correlation. If None, the internal reco-level or truth-level weights are used depending on the variables

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        ValueError
            If the variables are not all of reco level or truth level
        """

        if weights is None:
            isReco = np.all([self._in_data_reco(var) for var in variables])
            isTrue = np.all([self._in_data_truth(var) for var in variables])
            if isReco:
                w = self.get_weights(reco_level=True)
            elif isTrue:
                w = self.get_weights(reco_level=False)
            else:
                raise ValueError(f"Variables are unknown or are a mixture of reco- and truth-level variables: {variables}")
        else:
            w = weights

        cor_df = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)

        for var1, var2 in util.pairwise(variables):
            cor12 = util.cor_w(self[var1], self[var2], w)
            cor_df[var1][var2] = cor12
            cor_df[var2][var1] = cor12

        return cor_df

    def get_histogram(self, variable, bin_edges, weights=None, density=False, norm=None, absoluteValue=False, extra_cuts=None):
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
        density : bool
            If True, normalize the histogram by bin widths
        norm : float, default None
            If not None, rescale the histogram to norm
        absoluteValue : bool
            If True, fill the histogram with the absolute value
        extra_cuts : array-like of shape (nevents,) of bool, default None
            An array of flags to select events that are included in filling the histogram

        Returns
        -------
        A Hist object if `weights` is 1D, a list of Hist objects if `weights` 
            is 2D.
        """
        if weights is None:
            if self._in_data_truth(variable): # mc truth level
                weights = self.get_weights(reco_level=False)
            else: # reco level
                weights = self.get_weights(reco_level=True)

        if isinstance(weights, np.ndarray):
            if weights.ndim == 1: # if weights is a 1D array
                # make histogram with valid events
                varr = self[variable]
                if absoluteValue:
                    varr = np.abs(varr)
                assert(len(varr) == len(weights))

                if extra_cuts is not None: # filter events
                    assert(len(varr) == len(extra_cuts))
                    return calc_hist(varr[extra_cuts], weights=weights[extra_cuts], bins=bin_edges, density=density, norm=norm)
                else:
                    return calc_hist(varr, weights=weights, bins=bin_edges, density=density, norm=norm)

            elif weights.ndim == 2: # make the 2D array into a list of 1D array
                return self.get_histogram(variable, bin_edges=bin_edges, weights=list(weights), density=density, norm=norm, absoluteValue=absoluteValue, extra_cuts=extra_cuts)
            else:
                raise RuntimeError("Only 1D or 2D array or a list of 1D array of weights can be processed.")
        elif isinstance(weights, list): # if weights is a list of 1D array
            hists = []
            for w in weights:
                h = self.get_histogram(variable, bin_edges=bin_edges, weights=w, density=density, norm=norm, absoluteValue=absoluteValue, extra_cuts=extra_cuts)
                hists.append(h)
            return hists
        else:
            raise RuntimeError("Unknown type of weights: {}".format(type(weights)))

    def get_histogram2d(
        self,
        variable_x,
        variable_y,
        bins_x,
        bins_y,
        weights=None,
        absoluteValue_x=False,
        absoluteValue_y=False,
        density=False
        ):
        """

        """
        varr_x = self.get_arrays(variable_x, valid_only=False)
        sel_x = self.pass_truth if self._in_data_truth(variable_x) else self.pass_reco

        varr_y = self.get_arrays(variable_y, valid_only=False)
        sel_y = self.pass_truth if self._in_data_truth(variable_y) else self.pass_reco

        sel = sel_x & sel_y
        varr_x = varr_x[sel]
        varr_y = varr_y[sel]

        if absoluteValue_x:
            varr_x = np.abs(varr_x)

        if absoluteValue_y:
            varr_y = np.abs(varr_y)

        if weights is None:
            w = self.get_weights(reco_level=True, valid_only=False)
            w = w[sel]
        elif len(weights) == len(sel):
            w = weights[sel]

        assert(len(varr_x) == len(w))
        assert(len(varr_y) == len(w))

        return calc_hist2d(varr_x, varr_y, bins=(bins_x, bins_y), weights=w, density=density)

    def get_histograms_flattened(
        self,
        variables, # list of str
        bins_dict,
        weights,
        density=False,
        norm=None,
        absoluteValues=False,
        extra_cuts=None
        ):

        if not isinstance(absoluteValues, list):
            absoluteValues = [absoluteValues] * len(variables)

        data_arrs = [
            np.abs(self[vname]) if absolute else self[vname]
              for vname, absolute in zip(variables, absoluteValues)
            ]

        weights = np.asarray(weights)
        if weights.ndim == 1: # 1D array

            for arr in data_arrs:
                assert(len(arr)==len(weights))

            if extra_cuts is not None: # filter events
                for arr in data_arrs:
                    assert(len(arr)==len(extra_cuts))

                data_arrs = [arr[extra_cuts] for arr in data_arrs]
                weights = weights[extra_cuts]

            if len(variables) == 2:
                return fh.FlattenedHistogram2D.calc_hists(
                    *data_arrs,
                    binning_d = bins_dict,
                    weights=weights,
                    norm=norm,
                    density=density
                    )
            elif len(variables) == 3:
                return fh.FlattenedHistogram3D.calc_hists(
                    *data_arrs,
                    binning_d = bins_dict,
                    weights=weights,
                    norm=norm,
                    density=density
                    )
            else:
                raise RuntimeError(f"Dimension {len(variables)} flattened histograms currently not supported")
        elif weights.ndim == 2: # 2D array
            hists = []
            for warr in weights:
                hists.append(
                    self.get_histograms_flattened(
                        variables,
                        bins_dict,
                        warr,
                        density=density,
                        norm=norm,
                        absoluteValues=absoluteValues,
                        extra_cuts=extra_cuts
                    )
                )
        else:
            raise RuntimeError("Only 1D or 2D array or a list of 1D array of weights can be processed.")

    def reset_underflow_overflow_flags(self):
        self.underflow_overflow_reco = False
        self.underflow_overflow_truth = False

    def update_underflow_overflow_flags(self, varname, bins):
        try:
            varr = self.get_arrays(varname, valid_only=False)

            isflow = (varr < bins[0]) | (varr > bins[-1])
            #TODO: update this for higher dimensions
        except KeyError:
            isflow = False

        if self._in_data_reco(varname):
            self.underflow_overflow_reco |= isflow
        elif self._in_data_truth(varname):
            self.underflow_overflow_truth |= isflow

    def is_underflow_or_overflow(self):
        return self.underflow_overflow_reco | self.underflow_overflow_truth

    def clear_underflow_overflow_events(self):
        notflow = ~self.is_underflow_or_overflow()

        self.data_reco = self.data_reco[notflow]
        self.pass_reco = self.pass_reco[notflow]
        self.weights = self.weights[notflow]

        if self.data_truth is not None:
            self.data_truth = self.data_truth[notflow]
            self.pass_truth = self.pass_truth[notflow]
            self.weights_mc = self.weights_mc[notflow]

        self.reset_underflow_overflow_flags()

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
    """

    def __init__(self):
        self.data_reco = None
        self.data_truth = None
        self.pass_reco = None
        self.pass_truth = None
        self.weights = None
        self.weights_mc = None

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