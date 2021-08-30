"""
Functions to compute metrics for quantifying the unfolding performance
"""
import os
import numpy as np
from scipy import stats

import plotting
from util import prepend_arrays, write_dict_to_json
from histogramming import get_values_and_errors

def _compute_metrics_wrt_ref(
        metrics_algo,
        histograms,
        histogram_ref
    ):
    """
    Compare histograms in the sequence to a reference histogram

    Parameters
    ----------
    metrics_algo : function
        function to compute metrics
    histograms :  Hist objects, or list or nested list of Hist objects
        histograms to be compared to the reference
    ref : Hist object
        reference to be compared with histograms

    Returns
    -------
    list or nested list of metrics 
        Dimension is the same as histograms
    """
    metrics = []

    if isinstance(histograms, list):
        # iterate over entries of the list 
        for h in histograms:
            metrics.append(_compute_metrics_wrt_ref(metrics_algo, h, histogram_ref))
    else:
        metrics = metrics_algo(histograms, histogram_ref)

    return metrics

def _compute_metrics_wrt_prev(
        metrics_algo,
        histograms
    ):
    """
    Compare histogram with the one of the previous iteration

    Parameters
    ----------
    metrics_algo : function
        function to compute metrics
    histograms : list or nested list of Hist objects
        list or nested list of Hist objects

    Returns
    -------
    list or nested list of metrics
    """
    metrics = []

    assert(isinstance(histograms, list))

    # check the type of entries of the list histograms
    if isinstance(histograms[0], list):
        for hists in histograms:
            metrics.append(_compute_metrics_wrt_prev(metrics_algo, hists))
    else:
        for hcur, hprev in zip(histograms[1:], histograms[:-1]):
            metrics.append(metrics_algo(hcur, hprev))

    return metrics

######
# Chi2/NDF
def compute_Chi2(hist_obs, hist_exp):
    """
    Compute Chi2 and numbers of degree of freedom between two Hist objects
    """

    if hist_obs.size != hist_exp.size:
        raise RuntimeError("Input histograms are not of the same size")

    # number of degrees of freedom
    ndf = hist_exp.axes[0].size

    # get bin entries and errors
    obs, obs_err = get_values_and_errors(hist_obs)
    exp, exp_err = get_values_and_errors(hist_exp)

    # loop over each bin and compute chi2
    chi2 = 0.
    for o, e, oerr, eerr in zip(obs, exp, obs_err, exp_err):
        if o == 0 and e==0:
            ndf -= 1 # BAD histogram binning!
            continue
        chi2 += ((o-e)**2)/(oerr**2+eerr**2)

    return chi2, ndf

def compute_pvalue_Chi2(chi2, ndf):
    return 1. - stats.chi2.cdf(chi2, ndf)

def write_texts_Chi2(histogram_ref, histograms, labels):
    assert(len(histograms)==len(labels))
    texts = ["$\\chi^2$/NDF (p-value):"]

    for h, l in zip(histograms, labels):
        if h is None:
            continue
        chi2, ndf = compute_Chi2(h, histogram_ref)
        pval = compute_pvalue_Chi2(chi2, ndf)

        texts.append("{} = {:.3f}/{} ({:.3f})".format(l, chi2, ndf, pval))

    return texts

def write_metrics_Chi2(hists_unfolded, hist_truth):
    res = _compute_metrics_wrt_ref(compute_Chi2, hists_unfolded, hist_truth)
    # res is a list or nested list of tuples (chi2, ndf)

    # get the array of chi2
    chi2_arr = np.asarray(res)[..., 0]

    # get the array of ndf
    ndf_arr = np.asarray(res)[..., 1]

    m = {}
    m["iterations"] = list(range(chi2_arr.shape[-1]))
    m["chi2/ndf"] = (chi2_arr / ndf_arr).tolist()
    m["ndf"] = ndf_arr.tolist()
    m["pvalue"] = compute_pvalue_Chi2(chi2_arr, ndf_arr).tolist()

    return m

def write_metrics_Chi2_wrt_prev(hists_unfolded):
    res = _compute_metrics_wrt_prev(compute_Chi2, hists_unfolded)
    chi2_arr = np.asarray(res)[..., 0]
    ndf_arr = np.asarray(res)[..., 1]

    m = {
        "iterations" : list(range(1,chi2_arr.shape[-1]+1)),
        "chi2/ndf" : (chi2_arr/ndf_arr).tolist(),
        "ndf" : ndf_arr.tolist(),
        "pvalue" : compute_pvalue_Chi2(chi2_arr, ndf_arr).tolist()
        }

    return m

######
# Triangular discriminator
def compute_Delta(histogram_1, histogram_2):
    if histogram_1.size != histogram_2.size:
        raise RuntimeError("Input histograms are not of the same size")

    delta = 0.
    for p, q in zip(histogram_1.values(), histogram_2.values()):
        if p==0 and q==0:
            continue # BAD histogram binning!
        delta += ((p-q)**2)/(p+q)*0.5

    return delta

def write_texts_Delta(histogram_ref, histograms, labels):
    assert(len(histograms)==len(labels))
    #texts = ["Triangular discriminator ($\\times 10^{-3}$):"]
    texts = ["Triangular discriminator:"]

    for h, l in zip(histograms, labels):
        if h is None:
            continue
        d = compute_Delta(d, histogram_ref)
        texts.append("{} = {:.3f}".format(l, d))

    return texts

def write_metrics_Detla(hists_unfolded, hist_truth):
    res = _compute_metrics_wrt_ref(compute_Delta, hists_unfolded, hist_truth)

    # number of iterations
    niters = np.asarray(res).shape[-1]

    return {
        "iterations" : list(range(niters)),
        "delta": res
        }

def write_metrics_Delta_wrt_prev(hists_unfolded):
    res = _compute_metrics_wrt_prev(compute_Delta, hists_unfolded)

    # number of iterations
    niters = np.asarray(res).shape[-1]
    
    return {
        "iterations" : list(range(1,niters+1)),
        "delta": res
        }

######
# Bin errors
def write_metrics_BinErrors(hists_unfolded):
    values, errors = get_values_and_errors(hists_unfolded)

    relerrs = np.asarray(errors) / np.asarray(values)

    niters = relerrs.shape[-2]

    m = {
        "iterations" : list(range(niters)),
        "percentage" : relerrs.tolist()
        }

    return m

######
# Kolmogorov–Smirnov test
def ks_2samp_weighted(data1, data2, weights1, weights2):
    # Two sample Kolmogorov–Smirnov test with weighted data
    # scipy.stats.ks_2samp does not support sample weights yet
    # cf. https://github.com/scipy/scipy/issues/12315
    # The following implementation is based on the solution here:
    # https://stackoverflow.com/questions/40044375

    index1 = np.argsort(data1)
    index2 = np.argsort(data2)

    d1_sorted = data1[index1]
    d2_sorted = data2[index2]

    w1_sorted = weights1[index1]
    w2_sorted = weights2[index2]
    w1_sorted /= np.mean(w1_sorted)
    w2_sorted /= np.mean(w2_sorted)
    n1 = sum(w1_sorted)
    n2 = sum(w2_sorted)

    d_all = np.concatenate([d1_sorted, d2_sorted])

    cw1 = np.hstack([0, np.cumsum(w1_sorted)/n1])
    cw2 = np.hstack([0, np.cumsum(w2_sorted)/n2])

    cdf1_w = cw1[np.searchsorted(d1_sorted, d_all, side='right')]
    cdf2_w = cw2[np.searchsorted(d2_sorted, d_all, side='right')]

    ks = np.max(np.abs(cdf1_w - cdf2_w))

    en = np.sqrt(n1 * n2 / (n1 + n2))
    prob = stats.kstwobign.sf(ks * en)

    # TODO return "normalized" ks?
    # ks * en

    return ks, prob

def write_texts_KS(data_ref, weights_ref, data_list, weights_list, labels):
    assert(len(weights_list) == len(labels))
    texts = ["KS test (two-sided p-value):"] #["$D_{KS}$:"]

    for data, w, l in zip(data_list, weights_list, labels):
        ks, prob = ks_2samp_weighted(data_ref, data, weights_ref, w)

        texts.append("{} = {:.2e} ({:.3f})".format(l, ks, prob))

    return texts

def compute_metrics_KS(
        data_ref,
        weights_ref,
        data,
        weights_arrs
    ):
    """
    Compute Kolmogorov–Smirnov test statistics between two weighted samples

    Parameters
    ----------
    data_ref : ndarray 
        shape (n_events_ref,)
    weights_ref: ndarray
        event weights of data_ref
        shape (n_events_ref,)
    data : ndarray
        shape (n_events,)
    weights_arrs : ndarray
        event weights of data_arrs
        shape (n_resamples, n_iterations, n_events) or (n_iterations, n_events)

    Returns
    -------
    (ndarray, ndarray)
        The first one in the tuple is KS test statistics, the second is p-values
        Shapes of the arrays are either (n_resamples, n_iterations) if the shape
        of weights_arrs is (n_resamples, n_iterations, n_events), or
        (n_iterations,) if weights_arr shape is (n_iterations, n_events)
    """

    if weights_arrs.ndim == 1:
        return ks_2samp_weighted(data_ref, data, weights_ref, weights_arrs)
    else:
        ks_arr = []
        prob_arr = []

        for weights in weights_arrs:
            ks, prob = compute_metrics_KS(data_ref, weights_ref, data, weights)
            ks_arr.append(ks)
            prob_arr.append(prob)

        return np.asarray(ks_arr), np.asarray(prob_arr)

######
def _prepend_prior(hprior, histlist):
    """
    Add the prior histogram to the front of a histogram list
    """
    newhistlist = []

    # check if the type of entries of histlist is also a list
    if isinstance(histlist[0], list):
        for hl in histlist:
            newhistlist.append(_prepend_prior(hprior, hl))
    else:
        newhistlist = [hprior] + histlist

    return newhistlist

def write_all_metrics_binned(
        hists_unfolded,
        hist_prior,
        hist_truth
    ):
    """
    Evaluate unfolded distributions

    Parameters
    ----------
    hists_unfolded : list of Hist objects
        unfolded distributions
    hist_truth : Hist object
        truth distribution to compare to
    hist_prior : Hist object
        prior distribution

    Returns
    -------
    dict
    """
    metrics = dict()

    hists_all = _prepend_prior(hist_prior, hists_unfolded)

    #####
    # Chi2/NDF
    if hist_truth is not None:
        # With respect to the truth distribution
        metrics["Chi2"] = write_metrics_Chi2(hists_all, hist_truth)

    # With respect to the previous iteration
    metrics["Chi2_wrt_prev"] = write_metrics_Chi2_wrt_prev(hists_all)

    #####
    # Triangular discriminator
    if hist_truth is not None:
        metrics["Delta"] = write_metrics_Detla(hists_all, hist_truth)

    metrics["Delta_wrt_prev"] = write_metrics_Delta_wrt_prev(hists_all)

    #####
    # Bin errors of unfolded histograms
    metrics["BinErrors"] = {}
    metrics["BinErrors"] = write_metrics_BinErrors(hists_all)

    bin_edges = hist_truth.axes[0].edges
    metrics["BinErrors"]["bin edges"] = bin_edges


    return metrics

def write_all_metrics_unbinned(
        data_arr_truth,
        weights_truth,
        data_arr_sim,
        weights_prior,
        weights_unfolded
    ):

    metrics = dict()

    #####
    # Kolmogorov–Smirnov test
    ks_arr, pval_arr = compute_metrics_KS(
        data_arr_truth, weights_truth, data_arr_sim, weights_unfolded
    )

    # for prior
    ks_prior, pval_prior = compute_metrics_KS(
        data_arr_truth, weights_truth, data_arr_sim, weights_prior
        )

    ks_arr = prepend_arrays(ks_prior, ks_arr)
    pval_arr = prepend_arrays(pval_prior, pval_arr)

    metrics["KS"] = {
        "iterations" : list(range(ks_arr.shape[-1])),
        "ks" : ks_arr.tolist(),
        "pvalue" : pval_arr.tolist()
        }

    return metrics

def evaluate_all_metrics(variable, varConfig, bin_edges, of, ibu=None):
    """
    Compute and plot metrics for unfolded distributions

    Parameters
    ----------
    variable : str
        Name of the variable for evaluating
    varConfig : dict
        Dictionary that contains configurations for the variable. Here only used
        to extract the truth-level branch name for the variable
    bin_edges : array-like of shape (nbins + 1,)
        Bin edges of the histogram for the variable,
    of : OmniFoldwBkg object
        OmniFold unfolder that contains data arrays and unfolded weights
    ibu : IBU object, optional
        IBU unfolder that provides unfolded distribution of the variable
        using the IBU method

    Returns
    -------
    A dictionary that contains the calculated metrics for various methods.
    The dictionary is also saved as a json file to of.outdir+"/Metrics/"
    """

    if not of.truth_known:
        print("Cannot evaluate the performance without a reference")
        return

    # truth-level branch name for variable
    vname_mc = varConfig['branch_mc']

    metrics_all = dict()
    metrics_all[variable] = {}

    # prior distribution
    hist_prior = of.datahandle_sig.get_histogram(vname_mc, bin_edges)

    # truth distribution
    hist_truth = of.datahandle_obs.get_histogram(vname_mc, bin_edges)

    ######
    # unfolded distribution from OmniFold
    hists_uf = of.get_unfolded_distribution(vname_mc, bin_edges, all_iterations=True)[0]

    metrics_all[variable]["nominal"] = write_all_metrics_binned(
        hists_uf, hist_prior, hist_truth)

    # data arrays of vname_mc
    # pseudo data
    data_truth = of.datahandle_obs[vname_mc]
    weights_truth = of.datahandle_obs.get_weights()
    # simulated data
    data_gen = of.datahandle_sig[vname_mc]
    weights_prior = of.datahandle_sig.get_weights()
    # unfolded weights
    weights_unfolded = weights_prior * of.unfolded_weights

    metrics_all[variable]["nominal"].update(
        write_all_metrics_unbinned(data_truth, weights_truth, data_gen,
                                   weights_prior, weights_unfolded)
        )

    # IBU for comparison if available
    if ibu is not None:
        hists_ibu = ibu.get_unfolded_distribution(all_iterations=True)[0]
        metrics_all[variable]["IBU"] = write_all_metrics_binned(
            hists_ibu, hist_prior, hist_truth)

    ######
    # resamples
    if of.unfolded_weights_resample is not None:
        hists_uf_resample = of.get_unfolded_hists_resample(
            vname_mc, bin_edges, all_iterations=True, normalize=True)

        metrics_all[variable]["resample"] = write_all_metrics_binned(
            hists_uf_resample, hist_prior, hist_truth)

        # If want to compute KS for every resample and every iteration
        # unfolded weights
        #weights_unfolded_rs = weights_prior * of.unfolded_weights_resample
        #metrics_all[variable]["resample"].update(
        #    write_all_metrics_unbinned(data_truth, weights_truth, data_gen,
        #                               weights_prior, weights_unfolded_rs)
        #    )

    ##########
    # write to json file
    outdir = os.path.join(of.outdir, "Metrics")
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    outname = os.path.join(outdir, variable+'.json')
    write_dict_to_json(metrics_all, outname)

    ##########
    # plot
    plot_all_metrics(metrics_all, variable, outdir)

    return metrics_all

################
# Plotting
def plot_all_metrics(metrics_dict, varname, outdir):
    """
    Plot metrics from the dictionary
    """
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    #####
    # Chi2 vs iterations
    # With respect to truth
    m_Chi2_OF = metrics_dict[varname]['nominal']['Chi2']
    m_Chi2_IBU = metrics_dict[varname]['IBU']['Chi2']

    figname = os.path.join(outdir, varname+'_Chi2s_wrt_Truth')
    plotting.plot_graphs(
        figname,
        [(m_Chi2_OF['iterations'], m_Chi2_OF['chi2/ndf']),
         (m_Chi2_IBU['iterations'], m_Chi2_IBU['chi2/ndf'])],
        labels = ['MultiFold', 'IBU'],
        xlabel = 'Iteration',
        ylabel = '$\\chi^2$/NDF w.r.t. truth',
        markers = ['o', 'o']
    )

    # With respect to previous iteration
    m_Chi2_OF_prev = metrics_dict[varname]['nominal']['Chi2_wrt_prev']
    m_Chi2_IBU_prev = metrics_dict[varname]['IBU']['Chi2_wrt_prev']

    figname = os.path.join(outdir, varname+'_Chi2s_wrt_Prev')
    plotting.plot_graphs(
        figname,
        [(m_Chi2_OF_prev['iterations'], m_Chi2_OF_prev['chi2/ndf']),
         (m_Chi2_IBU_prev['iterations'], m_Chi2_IBU_prev['chi2/ndf'])],
        labels = ['MultiFold', 'IBU'],
        xlabel = 'Iteration',
        ylabel = '$\\Delta\\chi^2$/NDF',
        markers = ['*', '*']
    )

    # All resamples
    m_Chi2_OF_rs = metrics_dict[varname]['resample']['Chi2']
    figname = os.path.join(outdir, varname+'_AllResamples_Chi2s_wrt_Truth')

    dataarr = [(m_Chi2_OF_rs['iterations'], y) for y in m_Chi2_OF_rs['chi2/ndf']]
    plotting.plot_graphs(
        figname,
        dataarr,
        xlabel = 'Iteration',
        ylabel = '$\\chi^2$/NDF w.r.t. truth',
        markers = ['o'] * len(dataarr),
        lw = 0.7, ms = 0.7
    )

    #####
    # Triangular discriminator vs iterations
    # With respsect to truth
    m_Delta_OF = metrics_dict[varname]['nominal']['Delta']
    m_Delta_IBU = metrics_dict[varname]['IBU']['Delta']

    plotting.plot_graphs(
        os.path.join(outdir, varname+'_Delta_wrt_Truth'),
        [(m_Delta_OF['iterations'], m_Delta_OF['delta']),
         (m_Delta_IBU['iterations'], m_Delta_IBU['delta'])],
        labels = ['MultiFold', 'IBU'],
        xlabel = 'Iteration',
        ylabel = 'Triangular discriminator w.r.t. truth',
        markers = ['o', 'o']
    )

    # All resamples
    m_Delta_OF_rs = metrics_dict[varname]['resample']['Delta']

    dataarr = [(m_Delta_OF_rs['iterations'], y) for y in m_Delta_OF_rs['delta']]
    plotting.plot_graphs(
        os.path.join(outdir, varname+'_AllResamples_Delta_wrt_Truth'),
        dataarr,
        xlabel = 'Iteration',
        ylabel = 'Triangular discriminator w.r.t. truth',
        markers = ['o'] * len(dataarr),
        lw = 0.7, ms = 0.7
    )

    #####
    # Bin Errors
    m_BinErr_OF = metrics_dict[varname]['nominal']['BinErrors']
    nbins = len(m_BinErr_OF['bin edges']) - 1

    # Bin error vs iterations for each bin
    relerr_arr = [(
        m_BinErr_OF['iterations'],
        [ y[ibin] for y in m_BinErr_OF['percentage'] ] )
        for ibin in range(nbins)
    ]

    plotting.plot_graphs(
        os.path.join(outdir, varname+'_BinErrors'),
        relerr_arr,
        labels = ["bin {}".format(i) for i in range(1, nbins+1)],
        xlabel = 'Iteration',
        ylabel = 'Relative Bin Errors',
        markers = ['o'] * nbins,
        lw = 0.7, ms = 0.7
    )
