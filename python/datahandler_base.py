"""
Classes for working with OmniFold datasets.
"""

from collections.abc import Mapping
import numpy as np
import pandas as pd

from numpy.random import default_rng
rng = default_rng()

import util
from histUtils import calc_hist, calc_histnd
import FlattenedHistogram as fh

def filter_filepaths(file_paths, rescale_symbol='*'):
    """
    Check the list of file paths and extract the weight rescale factors

    Example:
    Input:
        file_paths = [file1.root*1,2, 0.5*file2.root, file3.root]
        rescale_symbol = '*'
    Return:
        [file1.root, file2.root, file3.root], [1.2, 0.5, 1.]
    """
    fpaths_new = []
    factors_renorm = []

    for fpath in file_paths:

        f_rescale = 1.

        fpath_components = fpath.split(rescale_symbol)
        for comp in fpath_components:
            try:
                f_rescale *= float(comp)
            except ValueError:
                fpaths_new.append(comp)

        factors_renorm.append(f_rescale)

    assert(len(fpaths_new)==len(file_paths))
    assert(len(fpaths_new)==len(factors_renorm))

    return fpaths_new, factors_renorm

# base class
class DataHandlerBase(Mapping):
    def __init__(self):
        # data array
        self.data_reco = None # reco level
        self.data_truth = None # truth level

        # event weights
        self.weights = None # reco level
        self.weights_mc = None # truth level

        # event selection flags
        self.pass_reco = None # reco level
        self.pass_truth = None # truth level

        # additional selection flag to filer events for response
        self.response_filter = None

        # overflow/underflow flags to be set later
        self.underflow_overflow_reco = False
        self.underflow_overflow_truth = False

        # scale factors for fluctuating event weights
        self.sf_bs = None

    def __len__(self):
        """
        Get the number of events in the dataset.

        Returns
        -------
        non-negative int
        """
        raise NotImplementedError

    def _get_reco_arr(self, feature):
        raise NotImplementedError

    def _get_truth_arr(self, feature):
        raise NotImplementedError

    def _get_reco_keys(self):
        raise NotImplementedError

    def _get_truth_keys(self):
        raise NotImplementedError

    def _filter_reco_arr(self, selections):
        raise NotImplementedError

    def _filter_truth_arr(self, selections):
        raise NotImplementedError

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
        if isinstance(variable, list):
            return all([self._in_data_reco(v) for v in variable])
        elif self.data_reco is None:
            return False
        else:
            return variable in self._get_reco_keys()

    def _in_data_truth(self, variable):
        if isinstance(variable, list):
            return all([self._in_data_truth(v) for v in variable])
        elif self.data_truth is None:
            return False
        else:
            return variable in self._get_truth_keys()

    def get_entries(self, reco_level=True, valid_only=True):
        if reco_level:
            if valid_only:
                return np.count_nonzero(self.pass_reco)
            else:
                return len(self.pass_reco)
        else:
            if valid_only:
                return np.count_nonzero(self.pass_truth)
            else:
                return len(self.pass_truth)

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
                return self._get_reco_arr(str(features))
            elif self._in_data_truth(features): # truth level
                return self._get_truth_arr(str(features))

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
            return iter(self._get_reco_keys())
        else:
            return iter(
                list(self._get_reco_keys()) +
                list(self._get_truth_keys())
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

        # bootstrap
        if bootstrap:
            # generate new weights only if reco_level or none has been generated before
            # this is to ensure the truth-level weights are fluctuated exactly the same  as the reco-level one
            if reco_level or self.sf_bs is None:
                # fluctuate event weights
                self.sf_bs = rng.poisson(1, size=len(weights))

            weights *= self.sf_bs

        if valid_only:
            weights = weights[sel]

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

    def compute_histogram(self, variables, bins, weights=None, density=False, norm=None, absoluteValue=False, extra_cuts=None, bootstrap=False):
        """
        Compute a histogram of weighted variable in the dataset

        Parameters
        ----------
        variables : str or list of str
            Name(s) of the variable(s) in the dataset to histogram
        bins : 1D array-like, a tuple/list of 1D array-like, or a dictionary
            Binning configurations of the histograms
        weights : array-like. Default None
            Array of per-event weights. The lowest dimension is of shape (nevents,). The outer dimensions determines the dimension of the retured array of histograms. If None, retrieve weights from the dataset.
        density : bool. Default False
            If True, normalize the histogram by bin widths
        norm : float. Default None
            If not None, rescale the histogram to norm
        absoluteValue : bool or a list of bool. Default False
            If True, fill the histogram with the absolute value. If a list, the number of elements needs to match the number of variables
        extra_cuts : array-like of shape (nevents,) of bool. Default None
            If not None, used as filters to select a subset of events for filling the histograms
        bootstrap : bool. Default False
            If True, multiply each weight by a random value drawn from a Poisson
            distribution with mean of one.

        Returns
        -------
        A histogram object or an array of histogram objects. The return shape is determined based on the outer dimensions of the weights array.
        If bins is an integer or a 1D array, the returned histograms are 1D hist.Hist.
        If bins is a tuple/list of an integer or array, the returned histograms are 2D/3D/...D hist.Hist
        If bins is a dictionary, the returned histograms are fh.FlattenedHistogram2D or fh.FlattenedHistogram3D
        """
        if isinstance(variables, str):
            variables = [variables]

        have_reco = any([self._in_data_reco(v) for v in variables])
        have_truth = any([self._in_data_truth(v) for v in variables])

        # event filters
        sel = True
        if have_reco:
            sel &= self.pass_reco
        if have_truth:
            sel &= self.pass_truth

        if extra_cuts is not None: # additional filters
            sel &= extra_cuts

        # weights
        if weights is None:
            weights = self.get_weights(bootstrap=False, reco_level=have_reco, valid_only=False)[sel]

        if np.asarray(weights).ndim > 1:
            # call compute_histogram recursively with the inner dimensions
            return [self.compute_histogram(
                        variables, bins, wgts,
                        density=density,
                        norm=norm,
                        absoluteValue=absoluteValue,
                        extra_cuts=extra_cuts,
                        bootstrap=bootstrap
                        )
                    for wgts in weights ]

        # weights.ndim == 1
        # in case the provided weights are not yet filtered
        if len(weights) == len(sel) and not all(sel):
            weights = weights[sel]

        if bootstrap:
            weights *= rng.poisson(1, size=len(weights))

        # data arrays
        if not isinstance(absoluteValue, list):
            absoluteValue = [absoluteValue] * len(variables)
        elif len(absoluteValue) == 1:
            absoluteValue = absoluteValue * len(variables)

        varr_list = []
        for vname, absolute in zip(variables, absoluteValue):
            varr = self.get_arrays(vname, valid_only=False)[sel]
            if absolute:
                varr = np.abs(varr)
            assert len(varr) == len(weights)
            varr_list.append(varr)

        common_args = {'weights': weights, 'density': density, 'norm': norm}

        # make histogram
        if not isinstance(bins, dict):
            if len(varr_list) == 1:
                # check if bins is an integer or 1D array
                assert np.isscalar(bins) or np.asarray(bins).ndim==1
                return calc_hist(varr_list[0], bins=bins, **common_args)
            else:
                assert all([np.isscalar(b) or np.asarray(b).ndim==1 for b in bins])
                return calc_histnd(varr_list, bins_list=bins, **common_args)
        else:
            # fh.FlattenedHistogram
            if len(varr_list) == 2:
                return fh.FlattenedHistogram2D.calc_hists(*varr_list, binning_d=bins, **common_args)
            elif len(varr_list) == 3:
                return fh.FlattenedHistogram3D.calc_hists(*varr_list, binning_d=bins, **common_args)
            else:
                raise RuntimeError(f"Unsupported number of variables for FlattenedHistogram")

    def get_histogram(self, variable, bin_edges, weights=None, density=False, norm=None, absoluteValue=False, extra_cuts=None, bootstrap=False):
        """
        For backward compatibility
        """
        return self.compute_histogram(
            variable,
            bin_edges,
            weights=weights,
            density=density,
            norm=norm,
            absoluteValue=absoluteValue,
            extra_cuts=extra_cuts,
            bootstrap=bootstrap)

    def get_histogram2d(
        self,
        variable_x,
        variable_y,
        bins_x,
        bins_y,
        weights=None,
        absoluteValue_x=False,
        absoluteValue_y=False,
        density=False,
        bootstrap=False
        ):
        """
        For backward compatibility
        """
        return self.compute_histogram(
            [variable_x, variable_y],
            [bins_x, bins_y],
            weights=weights,
            density=density,
            absoluteValue=[absoluteValue_x, absoluteValue_y],
            bootstrap=bootstrap
        )

    def get_response(
        self,
        variable_reco,
        variable_truth,
        bins_reco,
        bins_truth,
        absoluteValue=False,
        normalize_truthbins=True,
        weights = None,
        ):

        if isinstance(variable_reco, list):
            assert len(variable_reco) == 1
            variable_reco = variable_reco[0]

        if isinstance(variable_truth, list):
            assert len(variable_truth) == 1
            variable_truth = variable_truth[0]

        if not self._in_data_reco(variable_reco):
            raise ValueError(f"Array for variable {variable_reco} not available")
        elif not self._in_data_truth(variable_truth):
            raise ValueError(f"Array for variable {variable_truth} not available")
        else:
            response = self.compute_histogram(
                [variable_reco, variable_truth],
                [bins_reco, bins_truth],
                absoluteValue = absoluteValue,
                weights = weights,
                extra_cuts = self.response_filter
            )

            if normalize_truthbins:
                # normalize per truth bin to 1
                #response.view()['value'] = response.values() / response.project(1).values()
                #response.view()['value'] = response.values() / response.values().sum(axis=0)
                response_normed = np.zeros_like(response.values())
                np.divide(response.values(), response.values().sum(axis=0), out=response_normed, where=response.values().sum(axis=0)!=0)

                response.view()['value'] = response_normed

            return response

    def get_histograms_flattened(
        self,
        variables, # list of str
        bins_dict,
        weights=None,
        density=False,
        norm=None,
        absoluteValues=False,
        extra_cuts=None,
        bootstrap=False
        ):
        """
        For backward compatibility
        """
        return self.compute_histogram(
            variables,
            bins_dict,
            weights=weights,
            density=density,
            norm=norm,
            absoluteValue=absoluteValues,
            extra_cuts=extra_cuts,
            bootstrap=bootstrap
        )

    def get_response_flattened(
        self,
        variables_reco, # list of str
        variables_truth, # list of str
        bins_reco_dict,
        bins_truth_dict,
        absoluteValues=False,
        normalize_truthbins=True,
        weights = None
        ):

        if not isinstance(absoluteValues, list):
            absoluteValues = [absoluteValues] * len(variables_reco)

        if len(variables_reco) == 2:
            fh_reco = fh.FlattenedHistogram2D(bins_reco_dict, *variables_reco)
            fh_truth = fh.FlattenedHistogram2D(bins_truth_dict, *variables_truth)
        elif len(variables_reco) == 3:
            fh_reco = fh.FlattenedHistogram3D(bins_reco_dict, *variables_reco)
            fh_truth = fh.FlattenedHistogram3D(bins_truth_dict, *variables_truth)
        else:
            raise RuntimeError(f"Dimension {len(variables_reco)} flattened histograms currently not supported")

        fh_response = fh.FlattenedResponse(fh_reco, fh_truth)

        # event selections
        passall = self.pass_reco & self.pass_truth
        if self.response_filter is not None:
            passall &= self.response_filter

        # data arrays
        data_arr_reco = []
        for vname, absolute in zip(variables_reco, absoluteValues):
            varr_reco = self.get_arrays(vname, valid_only=False)
            varr_reco = varr_reco[passall]

            if absolute:
                varr_reco = np.abs(varr_reco)

            data_arr_reco.append(varr_reco)

        data_arr_truth = []
        for vname, absolute in zip(variables_truth, absoluteValues):
            varr_truth = self.get_arrays(vname, valid_only=False)
            varr_truth = varr_truth[passall]

            if absolute:
                varr_truth = np.abs(varr_truth)

            data_arr_truth.append(varr_truth)

        if weights is None:
            weight_arr = self.get_weights(reco_level=True, valid_only=False)
            weight_arr = weight_arr[passall]
        else:
            assert len(weights) == len(passall)
            weight_arr = weights[passall]

        fh_response.fill(data_arr_reco, data_arr_truth, weight=weight_arr)

        if normalize_truthbins:
            fh_response.normalize_truth_bins()

        return fh_response

    def remove_unmatched_events(self):
        # keep only events that pass all selections
        if self.data_truth is None:
            # reco only
            self._filter_reco_arr(self.pass_reco)
            self.weights = self.weights[self.pass_reco]
            self.pass_reco = self.pass_reco[self.pass_reco]
        else:
            pass_all = self.pass_reco & self.pass_truth
            self._filter_reco_arr(pass_all)
            self._filter_truth_arr(pass_all)
            self.weights = self.weights[pass_all]
            self.weights_mc = self.weights_mc[pass_all]

            self.pass_reco = self.pass_reco[pass_all]
            self.pass_truth = self.pass_truth[pass_all]

    def remove_events_failing_reco(self):
        if self.data_truth is not None:
            self._filter_truth_arr(self.pass_reco)
            self.weights_mc = self.weights_mc[self.pass_reco]
            self.pass_truth = self.pass_truth[self.pass_reco]

        self._filter_reco_arr(self.pass_reco)
        self.weights = self.weights[self.pass_reco]
        self.pass_reco = self.pass_reco[self.pass_reco]

    def remove_events_failing_truth(self):
        if self.data_truth is None:
            return

        self._filter_reco_arr(self.pass_truth)
        self.weights = self.weights[self.pass_truth]
        self.pass_reco = self.pass_reco[self.pass_truth]

        self._filter_truth_arr(self.pass_truth)
        self.weights_mc = self.weights_mc[self.pass_truth]
        self.pass_truth = self.pass_truth[self.pass_truth]

    def reset_underflow_overflow_flags(self):
        self.underflow_overflow_reco = False
        self.underflow_overflow_truth = False

    def update_underflow_overflow_flags(self, varnames, bins):
        try:
            varr = self.get_arrays(varnames, valid_only=False)

            if isinstance(bins, np.ndarray):
                isflow = (varr < bins[0]) | (varr > bins[-1])
            elif isinstance(bins, dict) and varr.shape[-1] == 2:
                fh2d = fh.FlattenedHistogram2D(bins)
                isflow = fh2d.is_underflow_or_overflow(varr[:,0], varr[:,1])
            elif isinstance(bins, dict) and varr.shape[-1] == 3:
                fh3d = fh.FlattenedHistogram3D(bins)
                isflow = fh3d.is_underflow_or_overflow(varr[:,0], varr[:,1], varr[:,2])
            else:
                raise RuntimeError(f"Cannot handle data array of shape {varr.shape} with binning config {bins}")
        except KeyError:
            isflow = False

        # for now: assume all varnames are either reco or truth variables
        if self._in_data_reco(varnames):
            self.underflow_overflow_reco |= isflow
        elif self._in_data_truth(varnames):
            self.underflow_overflow_truth |= isflow

    def is_underflow_or_overflow(self):
        return self.underflow_overflow_reco | self.underflow_overflow_truth

    def clear_underflow_overflow_events(self):
        notflow = not self.is_underflow_or_overflow()

        self._filter_reco_arr(notflow)
        self.pass_reco = self.pass_reco[notflow]
        self.weights = self.weights[notflow]

        if self.data_truth is not None:
            self._filter_truth_arr(notflow)
            self.pass_truth = self.pass_truth[notflow]
            self.weights_mc = self.weights_mc[notflow]

        self.reset_underflow_overflow_flags()

def filter_variable_names(variable_names):
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