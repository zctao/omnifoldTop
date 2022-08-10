"""
preprocessor class that is used to apply pre training mapping to data, gen, sim
preprocessor only uses trainig variables, extra observables do not enter training and are not preprocessed
the implementation of this class use the assumption that feature array is constructed in the same order as how the observable argument is ordered
remember to update this class if that assumption is changed

the preprocessor config should be written in two sections in the following format:
{
    "featrue": {
        "name of feature preprocessor": ["variables to apply to", "for example", "th_phi"],
        "there can be multiple of them": ["with", "same", "format"],
        "use all keyword to specify preprocessing all observables": ["all"]
    },
    "weight": ["a list", "of", "weight preprocessing", "function names"]
}
the preprocessor functions will be run in the same order as how they are supplied (top to bottom for features, left to right for weights)
"""
import json
import numpy as np
from collections import OrderedDict
import gc

import logging
logger = logging.getLogger('Preprocessor')
logger.setLevel(logging.DEBUG)

# keys in config
FEATURE = "feature"
WEIGHT = "weight"

# observable name indicating all observables will be preprocessed
all_observable_key = "all_observables"

# preprocessor class

class Preprocessor():
    def __init__(self, observable_dict, prep_config_path) -> None:
        """
        arguments
        ---------
        observable_dict: dictionary
            loaded from observable config
        prep_config_path: str
            path to the preprocessor config file, usually located in configs/preprocessor
        """

        logger.info("Initializing Preprocessor")

        # map from string options to functions

        self.feature_preprocessing_function_map = {
            "angle_to_sin_cos": self._angle_to_sin_cos,
            "angle_to_sin": self._angle_to_sin,
            "angle_to_cos": self._angle_to_cos,
            "divide_by_magnitude_of_main": self._divide_by_magnitude_of_mean,
            "standardize": self._standardize
        }

        weight_preprocessing_function_map = {

        }

        # convert observable dict to a variable dict, mapping branch name to observable name
        self.observable_name_dict = {}
        for ob_name in observable_dict:
            self.observable_name_dict[observable_dict[ob_name]["branch_det"]] = ob_name
            self.observable_name_dict[observable_dict[ob_name]["branch_mc"]] = ob_name

        # read in config, fill those without assigned preprocessor to an empty list
        with open(prep_config_path, "r") as config_file:
            self.config = json.load(config_file, object_pairs_hook=OrderedDict)

        self.weight_preprocessing_functions = [weight_preprocessing_function_map[name] for name in self.config[WEIGHT]]

        logger.debug("Initializing Preprocessor: Done")

    # preprocessor functions defined here
    # they should be of the form
    # arguments: feature_array, mask, observables, other keyword arguments
    # returns: preprocessed array, modified observable list (in this order)

    def _angle_to_sin_cos(self, feature_array, mask, observables, **args):
        """
        maps an angle to its sine and cosine

        arguments
        ---------
        feature_array: 2d numpy array
            feature array of the shape (number of events, number of observables)
        mask: list of boolean
            indicating which observables to be modified
        observables: numpy array of str
            observable names indicating their position in the feature array

        returns
        -------
        feature_array: 2d numpy array
            a 2d feature array with additional observable (and original angle replaced) in each event representing the sine and cosine value of the corresponding original angle measurement
        observables: numpy array of str
            a 1d str array representing the order of observables after this step of preprocessing
        """

        constant = feature_array[:, ~mask] # part of feature array that will not get modified
        modifying = feature_array[:, mask] # angles that will be encoded

        feature_array = np.concatenate((constant, np.sin(modifying), np.cos(modifying)), axis = 1)

        observables = np.concatenate((observables[~mask], observables[mask], observables[mask]))

        return feature_array, observables

    def _angle_to_sin(self, feature_array, mask, observables, **args):
        """
        maps an angle to sine of that angle

        arguments
        ---------
        feature_array: 2d numpy array 
            feature array of the shape (number of events, number of observables)
        mask: list of boolean
            indicating which observables to be modified
        observables: numpy array of str
            observable names indicating their position in the feature array

        returns
        -------
        feature array: 2d numpy array
            a 2d feature array with the original angle observable replaced by its sine value
        observables: numpy array of str
            a 1d str array representing the order of observables after this step of preprocessing
        """

        feature_array[:, mask] = np.sin(feature_array[:, mask])

        return feature_array, observables

    def _angle_to_cos(self, feature_array, mask, observables, **args):
        """
        maps an angle to cosine of that angle

        arguments
        ---------
        feature_array: 2d numpy array 
            feature array of the shape (number of events, number of observables)
        mask: list of boolean
            indicating which observables to be modified
        observables: numpy array of str
            observable names indicating their position in the feature array

        returns
        -------
        feature array: 2d numpy array
            a 2d feature array with the original angle observable replaced by its cosine value
        observables: numpy array of str
            a 1d str array representing the order of observables after this step of preprocessing
        """
        feature_array[:, mask] = np.cos(feature_array[:, mask])

        return feature_array, observables

    def _divide_by_magnitude_of_mean(self, feature_array, mask, observables, **args):
        """
        normalize the given feature array through dividing by the order of magnitude of the mean

        arguments
        ---------
        feature_array: numpy 2d array 
            feature array of shape (number of events, features in each event)
        mask: list of boolean
            indicating which observables to be modified
        observables: numpy array of str
            observable names indicating their position in the feature array

        returns
        -------
        feature_array: numpy 2d array
            feature array divided by its order of magnitude of the mean, should have its absolute value in the order of 1
        observables: numpy array of str
            a 1d str array representing the order of observables after this step of preprocessing
        """

        mean = np.mean(np.abs(feature_array[:, mask]), axis=0)
        oom = 10**(np.log10(mean).astype(int))
        feature_array[:, mask] = feature_array[:, mask] / oom
        return feature_array, observable

    def _standardize(self, feature_array):
        """
        standardize the given feature array

        arguments
        ---------
        feature_array: numpy 2d array 
            feature array of shape (number of events, features in each event)
        mask: list of boolean
            indicating which observables to be modified
        observables: numpy array of str
            observable names indicating their position in the feature array

        returns
        -------
        feature_array: numpy 2d array
            standardized feature array with mean of 0 and std of 1
        observables: numpy array of str
            a 1d str array representing the order of observables after this step of preprocessing
        """

        # compute mean over all feature_arrays together
        mean = np.mean(feature_array[:, mask], axis=0)
        std = np.std(feature_array, axis=0)
        feature_array[:, mask] = (feature_array[:, mask] - mean) / std

        return feature_array, observable
        
    # auxiliary functions for preprocessing
    def _mask(self, observables, modify):
        """
        return a mask indicating whether each index is to be included in the preprocessing

        arguments
        ---------
        observables: numpy array
            the list of observable names indicating position of the observables in the feature array
        modify: list of str
            the name of the observables that will be modified by the next preprocessor

        returns
        -------
        mask: 1d numpy array of boolean 
            indicating whether each index is to be modified
        """
        mask = np.zeros(np.shape(observables))
        if all_observable_key in modify:
            # indicating all observables should be preprocessed
            mask = mask + 1 # set every mask item to 1
        else:
            for ob_name in modify:
                mask[observables == ob_name] = 1
        mask = mask == 1
        return mask

    def preprocess(self, features, feature_array, **args):
        """
        preprocess feature_array by applying requested preprocessor functions

        arguments
        ---------
        features: list of str
            name of root tree branchs, for example, "PseudoTop_Reco_ttbar_m"
        feature_array: 2d numpy array 
            feature array with shape (number of events, number of observables in each event)
        args: extra parameters that will be passed directly to supported preprocessor functions

        returns
        -------
        feature array: 2d numpy array
            preprocessed feature array
        """
        # convert feature names to observable names
        observables = np.array([self.observable_name_dict[feature] for feature in features])

        # use as a checklist to mark the items that are done
        task_list = (self.config[FEATURE]).copy()

        logger.debug("Observable order before preprocessing: "+str(observables))

        while task_list:
            function_name, modifying = task_list.popitem(last = False)
            # replace modifying with all observables if keyword "all" is supplied
            if "all" in modifying:
                modifying = observables
            logger.debug("Applying preprocessing function " + function_name)
            function = self.feature_preprocessing_function_map[function_name]

            # create a mask for the next operation
            mask = self._mask(observables, modifying)

            # call preprocessor function, passing any additional args as is
            # modify args here to add in additional arguments passed to preprocessor function
            feature_array, observables = function(feature_array, mask, observables, **args)

            logger.debug("Observable order after preprocessing round: "+str(observables))

        gc.collect()
        
        # return the feature array after preprocessing
        return feature_array
    
    def preprocess_weight(self, feature_arrays, weights, observables, **args):
        """
        apply the list of weight preprocessors in sequence and return the final set of preprocessed weights

        arguments
        ---------
        feature_arrays: tuple of numpy arrays
            feature arrays in the order of [data, sim, gen]
        weights: tuple of numpy arrays
            event weights in the same order as feature arrays
        observables: list of str
            indicating the position of each observable in feature array
        """

        for function in self.weight_preprocessing_functions:
            weights = function(feature_arrays, weights, observables, **args)

        return weights

    """
    preprocessor function for weights
    they should have the form
    arguments: feature arrays (corresponding to the weight, a list of feature arrays), weights (list of weights), observables (indicating order in feature array), other keyword arguments
    returns: reweighed weight array
    """



# preprocessor instance

preprocessor = None

def initialize(observable_dict, prep_config_path):
    """
    create a preprocessor instance from given parameters

    arguments
    ---------
    observable_dict: dictionary 
        loaded from observable config
    prep_config_path: str
        path to the preprocessor config file, usually located in configs/preprocessor
    """
    global preprocessor
    preprocessor = Preprocessor(observable_dict, prep_config_path)

def get() -> Preprocessor:
    """
    returns
    -------
    preprocessor: Preprocessor
        the preprocessor instance
    """
    return preprocessor