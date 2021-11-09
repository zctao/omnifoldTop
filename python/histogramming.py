"""
Helper functions to handle Hist objects.
"""

import numpy as np
import pandas as pd

import hist
from hist import Hist
import functools
import operator

import logging
logger = logging.getLogger('Histogramming')
logger.setLevel(logging.DEBUG)

def get_bin_widths(h):
    return h.axes.widths

def get_bin_centers(h):
    return h.axes.centers

def rescale_hist(h, norm=1.):
    # !! underflow and overflow bins are not included in the calculation
    h = h * norm / h.sum()['value']
    return h

def get_hist_areas(h):
    # !! underflow and overflow bins are not included in the calculation
    bin_widths = get_bin_widths(h)
    areas = functools.reduce(operator.mul, bin_widths)
    return areas

def get_density(h):
    # !! underflow and overflow bins are not included in the calculation
    areas = get_hist_areas(h)
    return h / h.sum()['value'] / areas

def check_hist_flow(h, threshold_underflow=0.01, threshold_overflow=0.01):
    n_underflow = h[hist.underflow]['value']
    n_overflow = h[hist.overflow]['value']
    n_total = h.sum(flow=True)['value']

    if float(n_underflow/n_total) > threshold_underflow:
        logger.debug("Percentage of entries in the underflow bin: {}".format(float(n_underflow/n_total)))
        logger.warn("Number of entries in the underflow bin exceeds the threshold!")
        return False

    if float(n_overflow/n_total) > threshold_overflow:
        logger.debug("Percentage of entries in the overflow bin: {}".format(float(n_overflow/n_total)))
        logger.warn("Number of entries in the overflow bin exceeds the threshold!")
        return False

    return True

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

    if check_flow:
        # Warn if underflow or overflow bins have non-negligible entries    
        check_hist_flow(h)

    if density:
        h = get_density(h)

    if norm is not None:
        h = rescale_hist(h, norm=norm)

    return h

def calc_hist2d(data_x, data_y, bins, weights=None, density=False, norm=None, check_flow=True):
    h2d = Hist(hist.axis.Variable(bins[0]), hist.axis.Variable(bins[1]), storage=hist.storage.Weight())
    h2d.fill(data_x, data_y, weight=weights)

    if check_flow:
        # TODO: check_hist2d_flow(h2d)
        pass

    if density:
        h2d = get_density(h2d)

    if norm is not None:
        h2d = rescale_hist(h2d, norm=norm)

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
