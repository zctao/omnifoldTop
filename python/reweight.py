"""
Reweighter definitions and handling.

Attributes
----------
rw : dict
    Mapping of reweighting name to `Reweighter` object. To register a new
    reweighting, add it to `rw`.
"""

from random import seed
from random import randint as rand

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

import logging
logger = logging.getLogger('reweight')
logger.setLevel(logging.INFO)

@dataclass
class Reweighter:
    """
    A reweighting function and the variables it expects.

    Attributes
    ----------
    func : callable(np.ndarray(nevents, *variables.shape)) -> np.ndarray(nevents,)
        Reweighting function
    variables : array-like of str
        Features used in the reweighting
    """

    func: Callable[[npt.ArrayLike], np.ndarray]
    variables: npt.ArrayLike

def gaussian_bump(mtt):
    k = 0.5
    m0 = 800
    sigma = 100
    return 1 + k * np.exp(-(((mtt - m0) / sigma) ** 2))


def gaussian_tail(mtt):
    k = 0.5
    m0 = 2000
    sigma = 1000
    return 1 + k * np.exp(-(((mtt - m0) / sigma) ** 2))

def multivar_default_eight_linear_each_pos(data):
    oom = _getOrderOfMagnitude(data)
    return 1 + np.sum(data / oom, axis = 1)

def random_all_polynomial(data):
    seed("atlas")
    length = len(data[0])
    oom = _getOrderOfMagnitude(data)
    info_msg = "The polynomial used for reweight is: "
    
    new_weight = np.ones(len(data))

    for i in range(5):
        exponent = rand(0, 10)
        index = rand(0,length)
        info_msg += "+feature[{0}]^{1}/oom[{0}]^{1}".format(index,exponent)

        new_weight += [ 
            (event[index] ** exponent)/(oom[index] ** exponent) for event in data
        ]
    
    logger.info(info_msg)
    return new_weight

    


rw = {
    "linear_th_pt": Reweighter(lambda th_pt: 1 + th_pt / 800, "th_pt"),
    "gaussian_bump": Reweighter(gaussian_bump, "mtt"),
    "gaussian_tail": Reweighter(gaussian_tail, "mtt"),
    "multivar_defaults_linear": Reweighter(multivar_default_eight_linear_each_pos, ['th_pt', 'th_y', 'th_phi', 'th_e', 'tl_pt', 'tl_y', 'tl_phi', 'tl_e']),
    "rand_all_polynomial": Reweighter(random_all_polynomial, None)
}

# auxillary functions for multivar

def _getOrderOfMagnitude(data):
    """
    produce an array of the order of magnitude of each feature

    Returns
    -------
    a numpy ndarray with same dimension as a event
    """
    # get order of magnitude, mean is along vertical slices for 2d arrays
    mean = np.mean(np.abs(data), axis=0)
    return 10**(np.log10(mean).astype(int)) 