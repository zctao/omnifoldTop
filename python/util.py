import os
import numpy as np
import json
from scipy.optimize import curve_fit
import logging
import itertools
import argparse

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

def getFilesExtension(file_names):
    ext = None
    for fname in file_names:
        fext = os.path.splitext(fname)[-1]
        if ext is None:
            ext = fext
        else:
            # check if all file extensions are consistent
            if ext != fext:
                raise RuntimeError('Files do not have the same extensions')

    return ext

def configRootLogger(filename=None, level=logging.INFO):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=msgfmt, datefmt=datefmt)

    if filename:
        addFileHandler(logging.getLogger(), filename)

def addFileHandler(logger, filename):
    msgfmt = '%(asctime)s %(levelname)-7s %(name)-15s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    # check if the directory exists
    dirname = os.path.dirname(filename)
    nodir = not os.path.isdir(dirname)
    if nodir:
        os.makedirs(dirname)

    fhdr = logging.FileHandler(filename, mode='w')
    fhdr.setFormatter(logging.Formatter(msgfmt, datefmt))

    # remove old file handlers if there are any
    for hdr in logger.handlers:
        if isinstance(hdr, logging.FileHandler):
            logger.removeHandler(hdr)

    logger.addHandler(fhdr)
    if nodir:
        logger.info(f"Create directory {dirname}")

# JSON encoder for numpy array
# https://pynative.com/python-serialize-numpy-ndarray-into-json/
from json import JSONEncoder, JSONDecoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class ParseEnvDecoder(JSONDecoder):
    def __init__(self, **kwargs):
        # replace the object_hook with the customized one
        kwargs["object_hook"] = self.object_hook
        super().__init__(**kwargs)

    def object_hook(self, obj):
        for k, v in obj.items():
            if isinstance(v,str):
                obj[k] = os.path.expandvars(v)
            elif isinstance(v, list):
                obj[k] = [os.path.expandvars(iv) if isinstance(iv,str) else iv for iv in v]
        return obj

def read_dict_from_json(filename_json):
    jfile = open(filename_json, "r")
    try:
        jdict = json.load(jfile, cls=ParseEnvDecoder)
    except json.decoder.JSONDecodeError:
        jdict = {}

    jfile.close()
    return jdict

def write_dict_to_json(aDictionary, filename_json):
    jfile = open(filename_json, "w")
    json.dump(aDictionary, jfile, indent=4, cls=NumpyArrayEncoder)
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

def get_bins_dict(fname_bins):
    if not os.path.isfile(fname_bins):
        raise RuntimeError(f"Cannot access binning config file {fname_bins}")

    # read bin config file
    bins_d_raw = read_dict_from_json(fname_bins)
    bins_d = {}

    for k, v in bins_d_raw.items():
        if isinstance(v, list):
            bins_d[k] = np.asarray(v)
        elif isinstance(v, dict):
            if 'nbins' in v and 'xmin' in v and 'xmax' in v:
                # equal bins
                bins_d[k] = np.linspace(v['xmin'], v['xmax'], v['nbins']+1)
            else:
                # just use the dictionary
                bins_d[k] = v

    return bins_d

def gaus(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

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

def labels_for_dataset(array, label):
    """
    Make a label array for events in a dataset.

    Parameters
    ----------
    array : sequence
        Dataset of events.
    label : int
        Class label for events in the dataset.

    Returns
    -------
    np.ndarray of shape (nevents,)
        ndarray full of `label`
    """
    return np.full(len(array), label)

def prepend_arrays(element, input_array):
    """
    Add element to the front of the array

    Parameters
    ----------
    element : entry to be added to the array
        Its type should be the same as dtype of input_array
    input_array : ndarray
        if input_array.ndim > 1, element is duplicated and add to the front along
        the last axis

    Returns
    -------
    ndarray
    """
    input_array = np.asarray(input_array)

    # make an array from element that can be concatenated with input_array
    shape = list(input_array.shape)
    shape[-1] = 1
    element_arr = np.full(shape, element)

    return np.concatenate([element_arr, input_array], axis=-1)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def cov_w(x, y, w):
    return np.average( (x - np.average(x, weights=w)) * (y - np.average(y, weights=w)), weights=w)

def cor_w(x,y,w):
    return cov_w(x, y, w) / np.sqrt( cov_w( x, x, w) * cov_w( y, y, w) )

import tracemalloc

def reportMemUsage(logger):
    mcurrent, mpeak = tracemalloc.get_traced_memory()
    logger.debug(f"Current memory usage: {mcurrent*10**-6:.1f} MB; Peak usage: {mpeak*10**-6:.1f} MB")

    mtrace = tracemalloc.get_tracemalloc_memory()
    logger.debug(f"Memory used by tracemalloc: {mtrace*10**-6:.1f} MB")

class ParseEnvVar(argparse.Action):
    def __init__(self, option_strings, dest, default=None, **kwargs):

        if default is not None:
            if isinstance(default, str):
                default = os.path.expandvars(default)
            else:
                default = [os.path.expandvars(dv) for dv in default]

        super().__init__(option_strings, dest, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):

        if isinstance(values, str):
            values = os.path.expandvars(values)
        else:
            values = [os.path.expandvars(v) for v in values]

        setattr(namespace, self.dest, values)

def get_obs_label(observables, obsConfig_d):
    if isinstance(observables, list):
        return [get_obs_label(obs, obsConfig_d) for obs in observables]
    else:
        label = "$" + obsConfig_d[observables]['symbol'] + "$"

        if obsConfig_d[observables].get("unit"):
            label += " [" + obsConfig_d[observables]["unit"] + "]"

        return label

def get_diffXs_label(
    observable_list,
    obsConfig_d,
    isRelative=False,
    lumi_unit='pb'
    ):

    symbol_list = [obsConfig_d[obs]['symbol'] for obs in observable_list]
    unit_list = [obsConfig_d[obs]['unit'] for obs in observable_list]

    ndim = len(observable_list)

    label_xs = "d" if ndim == 1 else f"d^{ndim}"
    label_xs += "\\sigma_{t\\bar{t}} / "

    if ndim > 1:
        label_xs += "("

    for symbol in symbol_list:
        label_xs += f"d{symbol}"

    if ndim > 1:
        label_xs += ")"

    if isRelative:
        label_xs = "1/\\sigma_{t\\bar{t}} \cdot " + label_xs

    label_unit = ''
    for u in set(unit_list):
        if u:
            # count the occurrences
            p = unit_list.count(u)
            if p > 1:
                label_unit += f"{u}$^{p}$ "
            else:
                label_unit += f"{u} "
    label_unit = label_unit.rstrip()

    if isRelative:
        label_unit = f"[1/{label_unit}]"
    else:
        label_unit = f"[{lumi_unit}/{label_unit}]"

    label = "$" + label_xs + "$ " + label_unit
    return label