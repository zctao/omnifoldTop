import os
import numpy as np
import json
from scipy import stats
from scipy.optimize import curve_fit
from packaging import version
import tensorflow as tf
import logging

def parse_input_name(fname):
    fname_list = fname.split('*')
    if len(fname_list) == 1:
        return fname, 1.
    else:
        try:
            name = fname_list[0]
            rwfactor = float(fname_list[1])
        except ValueError:
            name = fname_list[1]
            rwfactor = float(fname_list[0])
        except:
            print('Unknown data file name {}'.format(fname))

        return name, rwfactor

def expandFilePath(filepath):
    filepath = filepath.strip()
    if not os.path.isfile(filepath):
        # try expanding the path in the directory of this file
        src_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(src_dir, filepath)

    if os.path.isfile(filepath):
        return os.path.abspath(filepath)
    else:
        return None

def configRootLogger(filename=None, level=logging.INFO):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=msgfmt, datefmt=datefmt)
    if filename:
        # check if the directory exists
        dirname = os.path.dirname(filename)
        nodir = not os.path.isdir(dirname)
        if nodir:
            os.makedirs(dirname)

        fhdr = logging.FileHandler(filename, mode='w')
        fhdr.setFormatter(logging.Formatter(msgfmt, datefmt))
        logging.getLogger().addHandler(fhdr)

        if nodir:
            logging.info("Create directory {}".format(dirname))

def configGPUs(gpu=None, limit_gpu_mem=False, verbose=0):
    assert(version.parse(tf.__version__) >= version.parse('2.0.0'))
    # tensorflow configuration
    # device placement
    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(verbose > 0)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        logging.error("No GPU found!")
        raise RuntimeError("No GPU found!")

    if gpu is not None:
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')

    # limit GPU memory growth
    if limit_gpu_mem:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g,True)

def get_fourvector_array(pt_arr, eta_arr, phi_arr, e_arr, padding=True):
    """
    Combine and format pt, eta, phi, and e arrays into array of four vectors

    Each of the input array can be either an array of float or array of arrays
    with a shape of (nevents,) 

    If padding is set to True, pad the output array into homogeneous ndarray 
    with zeros, i.e. each event will have the same number of four vectors 
    Otherwise the output array will be an array of objects
    """

    # Check if the input arrays are of the same dimension
    if (not (pt_arr.shape==eta_arr.shape and pt_arr.shape==phi_arr.shape and pt_arr.shape==e_arr.shape)):
        raise Exception("Input arrays have to be of the same dimension.")

    # Stack arrays
    v4_arr = np.stack([pt_arr, eta_arr, phi_arr, e_arr], axis=1)
    
    # Determine if input arrays are arrays of floats or objects
    if pt_arr.dtype != np.dtype('O'):
        # array of float i.e. one four vector per event
        return v4_arr
    else:
        # array of objects (arrays)
        # i.e. each event can contain various number of four vectors

        stackarr = lambda v : np.stack([v[0],v[1],v[2],v[3]], axis=1)
        
        if not padding:
            v4_arr_new = np.asarray([stackarr(v4) for v4 in v4_arr])
            return v4_arr_new
        else:
            nv4max = max(len(v4[0]) for v4 in v4_arr)
            v4_arr_new = np.asarray([np.concatenate((stackarr(v4), np.zeros([nv4max-len(v4[0]),4]))) for v4 in v4_arr])
            return v4_arr_new


def prepare_data_omnifold(ntuple, padding=True):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (n_events, n_particles, 4)
    """

    # Total number of events
    nevents = len(ntuple)

    # TODO: check if ntuple['xxx'] exsits
    # jets
    jetv4 = get_fourvector_array(ntuple['jet_pt'], ntuple['jet_eta'], ntuple['jet_phi'], ntuple['jet_e'], padding=True)
    # shape: (nevents, NJetMax, 4)

    # lepton
    lepv4 = get_fourvector_array(ntuple['lep_pt'], ntuple['lep_eta'], ntuple['lep_phi'], ntuple['lep_e'], padding=True)
    # shape: (nevents, 4)

    # MET
    metv4 = get_fourvector_array(ntuple['met_met'], np.zeros(nevents), ntuple['met_phi'], ntuple['met_met'], padding=True)
    # shape: (nevent, 4)

    data = np.concatenate([np.stack([lepv4],axis=1), np.stack([metv4],axis=1), jetv4], axis=1)

    return data

def prepare_data_multifold(ntuple, variables, standardize=False, reshape1D=False):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (n_events, n_variables)
    """
    data = np.hstack([np.vstack( get_variable_arr(ntuple,var) ) for var in variables])

    if standardize:
        data = (data - np.mean(data, axis=0))/np.std(data, axis=0)

    if reshape1D and len(variables)==1:
        # In case only one variable, reshape a column array into a 1D array of the shape (n_events,)
        data = data.reshape(len(data))

    return data

def set_up_bins(xmin, xmax, nbins):
    bins = np.linspace(xmin, xmax, nbins+1)
    midbins = (bins[:-1]+bins[1:])/2
    binwidth = bins[1]-bins[0]

    return bins, midbins, binwidth

def normalize_histogram(bin_edges, hist, hist_unc=None):
    binwidths = bin_edges[1:] - bin_edges[:-1]
    norm = np.dot(hist, binwidths)
    hist /= norm
    if hist_unc is not None:
        hist_unc /= norm

def normailize_stacked_histogrms(bin_edges, hists, hists_unc=None):
    binwidths = bin_edges[1:] - bin_edges[:-1]
    hstacked = np.asarray(hists).sum(axis=0)
    norm = np.dot(hstacked, binwidths)

    if hists_unc is None:
        hists_unc = [None]*len(hists)
    else:
        assert(len(hists_unc)==len(hists))

    for h, herr in zip(hists, hists_unc):
        h /= norm
        if herr is not None:
            herr /= norm

def add_histograms(h1, h2=None, h1_err=None, h2_err=None, c1=1., c2=1.):
    hsum = c1*h1
    if h2 is not None:
        assert(len(h1)==len(h2))
        hsum += c2*h2

    sumw2 = np.zeros_like(h1)

    if h1_err is not None:
        assert(len(h1_err)==len(h1))
        sumw2 += (c1*h1_err)**2

    if h2_err is not None:
        assert(len(h2_err)==len(h2))
        sumw2 += (c2*h2_err)**2

    hsum_err = np.sqrt(sumw2)

    return hsum, hsum_err

def divide_histograms(h_numer, h_denom, h_numer_err=None, h_denom_err=None):
    ratio = np.divide(h_numer, h_denom, out=np.zeros_like(h_denom), where=(h_denom!=0))

    # bin errors
    if h_numer_err is None:
        h_numer_err = np.zeros_like(h_numer_err)
    if h_denom_err is None:
        h_denom_err = np.zeros_like(h_denom_err)

    #rerrsq = (h_numer_err**2 + h_denom_err**2 * ratio**2) / h_denom**2
    rerrsq = np.divide(h_numer_err**2 + h_denom_err**2 * ratio**2, h_denom**2, out=np.zeros_like(h_denom), where=(h_denom!=0))

    ratio_err = np.sqrt(rerrsq)

    return ratio, ratio_err

def compute_triangular_discr(histogram_1, histogram_2):
    if len(histogram_1) != len(histogram_2):
        raise RuntimeError("Input histograms are not of the same size")

    delta = 0.
    for p, q in zip(histogram_1, histogram_2):
        if p==0 and q==0:
            continue
        delta += ((p-q)**2)/(p+q)*0.5

    return delta * 1000

def write_triangular_discriminators(hist_ref, hists, labels):
    assert(len(hists)==len(labels))
    stamps = ["Triangular discriminator ($\\times 10^{-3}$):"]

    for h, l in zip(hists, labels):
        d = compute_triangular_discr(h, hist_ref)
        stamps.append("{} = {:.3f}".format(l, d))

    return stamps

def compute_chi2(hist_obs, hist_exp, hist_obs_err, hist_exp_err):
    assert(len(hist_exp)==len(hist_obs))
    assert(len(hist_exp)==len(hist_exp_err))
    assert(len(hist_obs)==len(hist_obs_err))
    ndf = len(hist_exp) # degree of freedom
    chi2 = 0.

    for o, e, oerr, eerr in zip(hist_obs, hist_exp, hist_obs_err, hist_exp_err):
        if o == 0 and e==0:
            ndf -= 1 # BAD histogram binning!
            continue
        chi2 += ((o-e)**2)/(oerr**2+eerr**2)

    return chi2, ndf

def compute_diff_chi2(histograms_arr, histograms_err_arr):
    # compute the chi2 per degree of freedom between each histogram and its neighboring one in a list
    diff_chi2s = []

    for h_current, h_previous, herr_current, herr_previous in zip(histograms_arr[1:], histograms_arr[:-1], histograms_err_arr[1:], histograms_err_arr[:-1]):
        chi2, ndf = compute_chi2(h_current, h_previous, herr_current, herr_previous)
        diff_chi2s.append(chi2/ndf)

    return diff_chi2s

def compute_diff_chi2_wrt_first(histograms_arr, histograms_err_arr):
    # compute the chi2 per degree of freedom between each histogram and the first one in the list
    diff_chi2s_vs_first = []
    hprior = histograms_arr[0]
    hprior_err = histograms_err_arr[0]
    for hist, hist_err in zip(histograms_arr, histograms_err_arr):
        chi2, ndf = compute_chi2(hist, hprior, hist_err, hprior_err)
        diff_chi2s_vs_first.append(chi2/ndf)

    return diff_chi2s_vs_first

def compute_pvalue(chi2, ndf):
    return 1 - stats.chi2.cdf(chi2, ndf)

def write_chi2(hist_ref, hist_ref_unc, hists, hists_unc, labels):
    assert(len(hists)==len(labels))
    stamps = ["$\\chi^2$/NDF (p-value):"]

    for h, herr, l in zip(hists, hists_unc, labels):
        if h is None:
            continue
        chi2, ndf = compute_chi2(h, hist_ref, herr, hist_ref_unc)
        pval = compute_pvalue(chi2, ndf)

        stamps.append("{} = {:.3f}/{} ({:.3f})".format(l, chi2, ndf, pval))

    return stamps

def read_dict_from_json(filename_json):
    jfile = open(filename_json, "r")
    try:
        jdict = json.load(jfile)
    except json.decoder.JSONDecodeError:
        jdict = {}

    jfile.close()
    return jdict

def write_dict_to_json(aDictionary, filename_json):
    jfile = open(filename_json, "w")
    json.dump(aDictionary, jfile, indent=4)
    jfile.close()

def get_bins(varname, fname_bins):
    if os.path.isfile(fname_bins):
        # read bins from the config
        binning_dict = read_dict_from_json(fname_bins)
        if varname in binning_dict:
            if isinstance(binning_dict[varname], list):
                return np.asarray(binning_dict[varname])
            elif isinstance(binning_dict[varname], dict):
                # equal bins
                return np.linspace(binning_dict[varname]['xmin'], binning_dict[varname]['xmax'], binning_dict[varname]['nbins']+1)
        else:
            pass
            print("  No binning information found is {} for {}".format(fname_bins, varname))
    else:
        print("  No binning config file {}".format(fname_bins))

    # if the binning file does not exist or no binning info for this variable is in the dictionary
    return None

def gaus(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

import math
def fit_gaussian_to_hist(histogram, binedges, dofit=True):

    midbins = (binedges[:-1] + binedges[1:]) / 2.

    # initial value
    mu0 = sum(midbins*histogram)/sum(histogram)
    sigma0 = np.sqrt(sum(histogram * (midbins - mu0)**2) / sum(histogram))
    a0 = max(histogram) #sum(histogram)/(math.sqrt(2*math.pi)*sigma0)
    par0 = [a0, mu0, sigma0]

    # fit
    if dofit:
        try:
            popt, pcov = curve_fit(gaus, midbins, histogram, p0=par0)

            A, mu, sigma = popt
            #A_err, mu_err, sigma_err = np.sqrt(np.diag(pcov))
        except RuntimeError as e:
            print("WARNING: fit failed with message: {}".format(e))
            print("Return initial value estimated from sample")
            A, mu, sigma = tuple(par0)
    else:
        A, mu, sigma = tuple(par0)

    return A, mu, sigma

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

    return ks, prob

def write_ks(data_ref, weights_ref, data_list, weights_list, labels):
    assert(len(data_list)==len(labels))
    stamps = ["KS test (two-sided p-value)"] #["$D_{KS}$:"]

    for data, w, l in zip(data_list, weights_list, labels):
        ks, prob = ks_2samp_weighted(data_ref, data, weights_ref, w)

        stamps.append("{} = {:.2e} ({:.3f})".format(l, ks, prob))

    return stamps
