"""
Classes for working with OmniFold datasets.
"""

from collections.abc import Mapping
import numpy as np
import pandas as pd

from numpy.random import default_rng
rng = default_rng()

import util
from histogramming import calc_hist, calc_hist2d
import FlattenedHistogram as fh

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
        return len(self.data_reco) if self.data_reco is not None else 0

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
        else:
            if self.data_reco is None:
                return False
            else:
                return variable in self.data_reco.dtype.names

    def _in_data_truth(self, variable):
        if isinstance(variable, list):
            return all([self._in_data_truth(v) for v in variable])
        else:
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

    def get_histogram(self, variable, bin_edges, weights=None, density=False, norm=None, absoluteValue=False, extra_cuts=None, bootstrap=False):
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
        bootstrap : bool, default: False
            Multiply each weight by a random value drawn from a Poisson
            distribution with lambda = 1.

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

                if bootstrap:
                    weights = weights * rng.poisson(1, size=len(weights))

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
        density=False,
        bootstrap=False
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

        if bootstrap:
            w = w * rng.poisson(1, size=len(weights))

        return calc_hist2d(varr_x, varr_y, bins=(bins_x, bins_y), weights=w, density=density)

    def get_response(
        self,
        variable_reco,
        variable_truth,
        bins_reco,
        bins_truth,
        absoluteValue=False,
        normalize_truthbins=True
        ):

        if not self._in_data_reco(variable_reco):
            raise ValueError(f"Array for variable {variable_reco} not available")
        elif not self._in_data_truth(variable_truth):
            raise ValueError(f"Array for variable {variable_truth} not available")
        else:
            response = self.get_histogram2d(
                variable_reco, variable_truth,
                bins_reco, bins_truth,
                absoluteValue_x=absoluteValue, absoluteValue_y=absoluteValue
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
        weights = None,
        density=False,
        norm=None,
        absoluteValues=False,
        extra_cuts=None,
        bootstrap=False
        ):

        if not isinstance(absoluteValues, list):
            absoluteValues = [absoluteValues] * len(variables)

        data_arrs = [
            np.abs(self[vname]) if absolute else self[vname]
              for vname, absolute in zip(variables, absoluteValues)
            ]

        if weights is None:
            if all([self._in_data_truth(v) for v in variables]): # mc truth level
                weights = self.get_weights(reco_level=False)
            else: # reco level
                weights = self.get_weights(reco_level=True)

        weights = np.asarray(weights)

        if weights.ndim == 1: # 1D array

            for arr in data_arrs:
                assert(len(arr)==len(weights))

            if bootstrap:
                weights = weights * rng.poisson(1, size=len(weights))

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

    def get_response_flattened(
        self,
        variables_reco, # list of str
        variables_truth, # list of str
        bins_reco_dict,
        bins_truth_dict,
        absoluteValues=False,
        normalize_truthbins=True
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

        weight_arr = self.get_weights(reco_level=True, valid_only=False)
        weight_arr = weight_arr[passall]

        fh_response.fill(data_arr_reco, data_arr_truth, weight=weight_arr)

        if normalize_truthbins:
            fh_response.normalize_truth_bins()

        return fh_response

    def remove_unmatched_events(self):
        # keep only events that pass all selections
        if self.data_truth is None:
            # reco only
            self.data_reco = self.data_reco[self.pass_reco]
            self.weights = self.weights[self.pass_reco]
            self.pass_reco = self.pass_reco[self.pass_reco]
        else:
            pass_all = self.pass_reco & self.pass_truth
            self.data_reco = self.data_reco[pass_all]
            self.data_truth = self.data_truth[pass_all]
            self.weights = self.weights[pass_all]
            self.weights_mc = self.weights_mc[pass_all]

            self.pass_reco = self.pass_reco[pass_all]
            self.pass_truth = self.pass_truth[pass_all]

    def remove_events_failing_reco(self):
        if self.data_truth is not None:
            self.data_truth = self.data_truth[self.pass_reco]
            self.weights_mc = self.weights_mc[self.pass_reco]
            self.pass_truth = self.pass_truth[self.pass_reco]

        self.data_reco = self.data_reco[self.pass_reco]
        self.weights = self.weights[self.pass_reco]
        self.pass_reco = self.pass_reco[self.pass_reco]

    def remove_events_failing_truth(self):
        if self.data_truth is None:
            return

        self.data_reco = self.data_reco[self.pass_truth]
        self.weights = self.weights[self.pass_truth]
        self.pass_reco = self.pass_reco[self.pass_truth]

        self.data_truth = self.data_truth[self.pass_truth]
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
        notflow = ~self.is_underflow_or_overflow()

        self.data_reco = self.data_reco[notflow]
        self.pass_reco = self.pass_reco[notflow]
        self.weights = self.weights[notflow]

        if self.data_truth is not None:
            self.data_truth = self.data_truth[notflow]
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