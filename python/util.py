import os
import numpy as np
import json
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

def reportGPUMemUsage(logger):
    gpus = tf.config.experimental.list_logical_devices('GPU')

    for device in gpus:
        info_d = tf.config.experimental.get_memory_info(device.name)
        logger.info(f"{device.name} mem usage: current = {info_d['current']*1e-6:.2f} MB peak = {info_d['peak']*1e-6:.2f} MB");

# JSON encoder for numpy array
# https://pynative.com/python-serialize-numpy-ndarray-into-json/
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

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
