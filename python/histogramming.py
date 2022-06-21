"""
Helper functions to handle Hist objects.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

import uproot
import hist
from hist import Hist
import functools
import operator

import logging
logger = logging.getLogger('Histogramming')
logger.setLevel(logging.DEBUG)

def get_bin_centers(h):
    return h.axes.centers

def get_bin_edges(h):
    return h.axes.edges

def get_bin_widths(h):
    return h.axes.widths

def get_hist_widths(h):
    """
    For 1D Histogram, this returns an array of bin widths
    For 2D Histogram, this returns a 2D array of bin areas
    Note: underflow and overflow bins are not included
    """
    return functools.reduce(operator.mul, h.axes.widths)

def integral(h, flow=False):
    """
    return the sum of entry*width of all bins
    """
    total = (h.values() * get_hist_widths(h)).sum()

    if flow: # include underflow and overflow bins
        total = total + h[hist.underflow]['value'] + h[hist.overflow]['value']

    return total

def get_hist_norm(h, density=False, flow=True):
    if density:
        return integral(h, flow=flow)
    else:
        return h.sum(flow=flow)['value']

def check_hist_flow(h, threshold_underflow=0.01, threshold_overflow=0.01, density=False):
    """
    Check if the underflow and overflow bins are above the thresholds
    """
    n_underflow = h[hist.underflow]['value']
    n_overflow = h[hist.overflow]['value']
    n_total = get_hist_norm(h, density=density, flow=True)

    if float(n_underflow/n_total) > threshold_underflow:
        logger.debug("Percentage of entries in the underflow bin: {}".format(float(n_underflow/n_total)))
        logger.warn("Number of entries in the underflow bin exceeds the threshold!")
        return False

    if float(n_overflow/n_total) > threshold_overflow:
        logger.debug("Percentage of entries in the overflow bin: {}".format(float(n_overflow/n_total)))
        logger.warn("Number of entries in the overflow bin exceeds the threshold!")
        return False

    return True

def renormalize_hist(h, norm=1., density=False, flow=True):
    old_norm = get_hist_norm(h, density=density, flow=flow)
    h *= norm / old_norm
    return h

def calc_hist(data, bins=10, weights=None, density=False, norm=None, check_flow=True):
    if np.ndim(bins) == 1: # an array
        bin_edges = np.asarray(bins)
    elif np.ndim(bins) == 0: # integer
        xmin = np.asarray(data).min()
        xmax = np.asarray(data).max()
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        bin_edges = np.linspace(xmin, xmax, bins+1)

    h = Hist(hist.axis.Variable(bin_edges), storage=hist.storage.Weight())
    h.fill(data, weight=weights)

    if density:
        # normalize by bin widths
        h /= get_hist_widths(h)

    if check_flow:
        # Warn if underflow or overflow bins have non-negligible entries    
        check_hist_flow(h, density=density)

    if norm is not None:
        h = renormalize_hist(h, norm=norm, density=density, flow=True)

    return h

def calc_hist2d(data_x, data_y, bins, weights=None, density=False, norm=None, check_flow=True):
    h2d = Hist(hist.axis.Variable(bins[0]), hist.axis.Variable(bins[1]), storage=hist.storage.Weight())
    h2d.fill(data_x, data_y, weight=weights)

    if check_flow:
        # TODO: check_hist2d_flow(h2d)
        pass

    if density:
        # normalize by bin widths
        h2d /= get_hist_widths(h2d)

    if norm is not None:
        h2d = renormalize_hist(h2d, norm=norm, density=density, flow=True)

    return h2d

def set_hist_contents(histograms, contents):
    #TODO: underflow and overflow?
    # histograms: hist  object or a list of hist objects
    # contents: a ndarray or a list/array of arrays
    assert(len(np.asarray(histograms)) == len(contents))
    if isinstance(histograms, list):
        for h, c in zip(histograms, contents):
            set_hist_contents(h, c)
    else:
        histograms.view()['value'] = contents

def set_hist_errors(histograms, errors):
    #TODO: underflow and overflow?
    # histograms: hist object or a list of hist objects
    # errors: a ndarray or a list/array of arrays
    assert(len(np.asarray(histograms)) == len(errors))
    if isinstance(histograms, list):
        for h, err in zip(histograms, errors):
            set_hist_errors(h, err)
    else:
        histograms.view()['variance'] = errors**2

def get_hist(bins, contents, errors=None):
    #TODO: underflow and overflow?
    h = Hist(hist.axis.Variable(bins), storage=hist.storage.Weight())
    set_hist_contents(h, contents)
    if errors is not None:
        set_hist_errors(h, errors)
    return h

def get_values_and_errors(histogram):
    if isinstance(histogram, list):
        hvals, herrs = [], []
        for h in histogram:
            hv, he = get_values_and_errors(h)
            hvals.append(hv)
            herrs.append(he)
        return hvals, herrs
    else:
        hval = histogram.values()
        herr = np.sqrt(histogram.variances()) if histogram.variances() is not None else np.zeros_like(hval)
        return hval, herr

def get_mean_from_hists(histogram_list):
    """
    Get mean of each bin from a list of histograms.

    Parameters:
    ----------
    histogram_list: a list of histogram objects.
        The shape of np.asarray(histogram_list) is expected to be either (n_resamples, n_bins) or (n_resamples, n_iterations, n_bins)

    Returns
    -------
    np.ndarray
        The shape of the array is either (n_bins,) or (n_iterations, n_bins)
    """
    if len(histogram_list) == 0: # in case it is empty
        return None
    else:
        return np.mean(np.asarray(histogram_list)['value'], axis=0)

def get_sigma_from_hists(histogram_list):
    """
    Get standard deviation of each bin from a list of histograms

    Parameters:
    ----------
    histogram_list: a list of histogram objects.
        The shape of np.asarray(histogram_list) is expected to be either (n_resamples, n_bins) or (n_resamples, n_iterations, n_bins)

    Returns
    -------
    np.ndarray
        The shape of the array is either (n_bins,) or (n_iterations, n_bins)
    """
    if len(histogram_list) == 0: # in case it is empty
        return None
    else:
        return np.std(np.asarray(histogram_list)['value'], axis=0, ddof=1)

def average_histograms(histograms_list):
    if not histograms_list:
        return None

    elif len(histograms_list) == 1:
        # only one histogram in the list
        return histograms_list[0]

    else:
        # take the hist from the first in the list
        h_result = histograms_list[0].copy()

        # mean of each bin
        hmean = get_mean_from_hists(histograms_list)

        # standard deviation of each bin
        hsigma = get_sigma_from_hists(histograms_list)

        # standard error of the mean
        hstderr = hsigma / np.sqrt( len(histograms_list) )

        # set the result histogram
        set_hist_contents(h_result, hmean)
        set_hist_errors(h_result, hstderr)

        return h_result

def get_variance_from_hists(histograms_list):
    histograms_arr = np.asarray(histograms_list)['value']

    # number of entries per bin
    N = histograms_arr.shape[0]

    # variance
    hvar = stats.moment(histograms_arr, axis=0, moment=2) * N / (N-1)
    #assert( np.all( np.isclose(hvar, np.var(histograms_arr, axis=0, ddof=1)) ) )

    # variance of sample variance
    # https://math.stackexchange.com/questions/2476527/variance-of-sample-variance
    hvar_mu4 = stats.moment(histograms_arr, axis=0, moment=4)
    hvar_mu2 = stats.moment(histograms_arr, axis=0, moment=2)
    hvar_var = hvar_mu4 / N - hvar_mu2*hvar_mu2 * (N-3) / N / (N-1)

    return hvar, hvar_var

def get_bin_correlations_from_hists(histogram_list):
    """
    Get bin correlations from a list of histograms

    Parameters:
    ----------
    histogram_list: a list of histogram objects.
        The shape of np.asarray(histogram_list) is expected to be either (n_resamples, n_bins) or (n_resamples, n_iterations, n_bins)

    Returns
    -------
    DataFrame correlation matrix or a list of them
    """
    if len(histogram_list) == 0: # in case it is empty
        return None

    hists_content = np.asarray(histogram_list)['value']
    if hists_content.ndim == 2:
        # input shape: (n_resamples, n_bins)
        df_i = pd.DataFrame(hists_content)
        bins_corr = df_i.corr()
        return bins_corr

    elif hists_content.ndim == 3: # all iterations
        # input shape: (n_resamples, n_iterations, n_bins)
        niters = hists_content.shape[1]
        bins_corr = []
        for i in range(niters):
            df_i = pd.DataFrame(hists_content[:,i,:])
            bins_corr.append(df_i.corr()) # ddof = 1
        return bins_corr
    else:
        logger.error("Cannot handle the histogram collection of dimension {}".format(hists_content.ndim))
        return None

def divide(h1, h2):
    #FIXME: deal with errors
    # cf. TH1::Divide in ROOT, both Poisson and Binomial errors

    ratios = h1.values() / h2.values()

    # Uncorrelated for now. Not a valid assumption if compute efficiency
    r_variance = (h1.variances() / h1.values()**2 + h2.variances() / h2.values()**2) * ratios**2

    hr = h1.copy()
    hr.view()['value'] = ratios
    hr.view()['variance'] = r_variance

    return hr

##
# utilities to write/read histograms to/from files
# Write
def write_dict_uproot(file_to_write, obj_dict, top_dir=''):
    for k, v in obj_dict.items():
        if isinstance(v, dict):
            write_dict_uproot(
                file_to_write, v, os.path.join(top_dir, k)
                )
        else:
            if isinstance(v, list):
                for iv, vv in enumerate(v):
                    file_to_write[os.path.join(top_dir, f"{k}-list-{iv}")] = vv
            else:
                file_to_write[os.path.join(top_dir, k)] = v

def write_histograms_dict_to_file(hists_dict, file_name):
    with uproot.recreate(file_name) as f:
        write_dict_uproot(f, hists_dict)

# Read
def fill_dict_from_path(obj_dict, paths_list, obj):
    if not paths_list:
        return

    p0 = paths_list[0]

    if len(paths_list) == 1:
        # list of objects are denoted by e.g. <obj_name>-list-0
        pp = p0.split('-list-')
        if len(pp) == 1: # does not contain '-list-'
            obj_dict[p0] = obj
        else: # should be added to list
            common_name = pp[0]
            if isinstance(obj_dict.get(common_name), list):
                obj_dict[common_name].append(obj)
            else:
                obj_dict[common_name] = [obj]
    else:
        if not p0 in obj_dict:
            obj_dict[p0] = {}

        fill_dict_from_path(obj_dict[p0], paths_list[1:], obj)

def read_histograms_dict_from_file(file_name):
    histograms_d = {}
    with uproot.open(file_name) as f:
        for k, v in f.classnames().items():
            if not v.startswith("TH"):
                continue

            # create nested dictionary based on directories
            paths = k.split(';')[0].split(os.sep)
            fill_dict_from_path(histograms_d, paths, f[k].to_hist())

    return histograms_d
