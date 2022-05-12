"""
Reweighter definitions and handling.

Attributes
----------
rw : dict
    Mapping of reweighting name to `Reweighter` object. To register a new
    reweighting, add it to `rw`.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt


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


rw = {
    "linear_th_pt": Reweighter(lambda th_pt: 1 + th_pt / 800, "th_pt"),
    "gaussian_bump": Reweighter(gaussian_bump, "mtt"),
    "gaussian_tail": Reweighter(gaussian_tail, "mtt"),
}

"""
TODO
take all measurables if needed - done
fix current gaussian and linear reweighter
interface for multivar reweight
implement multivar reweighter
normalization
"""