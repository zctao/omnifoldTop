"""
preprocessor class that is used to apply pre training mapping to data, gen, sim
preprocessor only uses trainig variables, extra observables do not enter training and are not preprocessed
the implementation of this class use the assumption that feature array is constructed in the same order as how the observable argument is ordered
remember to update this class if that assumption is changed

the preprocessor config should be written in three sections in the following format:
{
    "featrue": {
        "name of feature preprocessor": ["variables to apply to", "for example", "th_phi"],
        "there can be multiple of them": ["with", "same", "format"],
        "use all keyword to specify preprocessing all observables": ["all"]
    },
    "normalization": {
        "normalization method such as divide_by_magnitude_of_mean": ["all"]
    }
    "weight": ["a list", "of", "weight preprocessing", "function names"]
}
the preprocessor functions will be run in the same order as how they are supplied (top to bottom for features, left to right for weights)
for normalization preprocessors, the default expected observable configuration is "all". The preprocessor will try to optimize memory usage with that setting. 
Therefore, in the case of all observables are to be normalized, use the "all" keyword.
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
NORMALIZATION = "normalization"
WEIGHT = "weight"

# observable name indicating all observables will be preprocessed
all_observable_key = "all_observables"

# preprocessor class

class Preprocessor():
    def __init__(self, observable_dict, prep_config_path, default_observable_names) -> None:
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
            "angle_to_cos": self._angle_to_cos
        }

        self.normalizer_map = {
            "divide_by_magnitude_of_mean": DivideByMeansMagnitude,
            "standardize": Standardize
        }

        weight_preprocessing_function_map = {
            "standardize": self.standardize_weight
        }

        # convert observable dict to a variable dict, mapping branch name to observable name
        self.observable_name_dict = {}
        for ob_name in observable_dict:
            self.observable_name_dict[observable_dict[ob_name]["branch_det"]] = ob_name
            self.observable_name_dict[observable_dict[ob_name]["branch_mc"]] = ob_name

        # read in config, using OrderedDict to ensure order is correct
        with open(prep_config_path, "r") as config_file:
            self.config = json.load(config_file, object_pairs_hook=OrderedDict)

        
        if self.config[WEIGHT]: logger.debug("Weight preprocessors: "+str(self.config[WEIGHT]))
        self.weight_preprocessing_functions = [weight_preprocessing_function_map[name] for name in self.config[WEIGHT]]

        self.default_observable_names = np.array([self.observable_name_dict[observable] for observable in default_observable_names])

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

    def feature_preprocess(self, feature_array, features=None, **args):
        """
        preprocess feature_array by applying requested preprocessor functions

        arguments
        ---------
        feature_array: 2d numpy array 
            feature array with shape (number of events, number of observables in each event)
        features: list of str
            name of root tree branchs, for example, "PseudoTop_Reco_ttbar_m"
        args: extra parameters that will be passed directly to supported preprocessor functions

        returns
        -------
        feature array: 2d numpy array
            preprocessed feature array
        observables: 1d numpy array
            the order of observables in feature array after all feature preprocessing steps
        """
        if features is None:
            if self.default_observable_names is None:
                raise ValueError("if observable list is not provided at feature preprocessing stage, default observable names needs to be set")
            observables = self.default_observable_names
        else:
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

            gc.collect()

            logger.debug("Observable order after preprocessing round: "+str(observables))

        gc.collect()
        
        # return the feature array after preprocessing
        return feature_array, observables
    
    def apply_normalizer(self, arr_data, arr_sim, arr_gen, observables, **args):
        """
        preprocess feature array by applying requested normalizer functions. Normalizers are things like standardizing or dividing by order of magnitude of mean.
        this is implemented in almost identical way to to feature_preprocess. the two main reasons to keep them separated are:
        1. normalization is almost always required 
        2. normalization is usually applied to the entire feature array, so there can be extra memory saving optimizations
        3. normalization and feature preprocessing can be quite distinct, for example, normalization never changes observable order and type in feature array

        arguments
        ---------
        arr_data: 2d numpy array 
            data array with shape (number of events, number of observables in each event)
        arr_sim: 2d numpy array 
            sim array with shape (number of events, number of observables in each event)
        arr_gen: 2d numpy array 
            gen array with shape (number of events, number of observables in each event) 
        observables: list of str
            name of observables, for example, "th_pt". normalization happens after feature preprocessing, so translated observable name is expected instead of branch names
        args: extra parameters that will be passed directly to supported preprocessor functions

        returns
        -------
        arr_data: 2d numpy array 
            data array with shape (number of events, number of observables in each event)
        arr_sim: 2d numpy array 
            sim array with shape (number of events, number of observables in each event)
        arr_gen: 2d numpy array 
            gen array with shape (number of events, number of observables in each event) 
        """
        # use as a checklist to mark the items that are done
        task_list = (self.config[NORMALIZATION]).copy()

        while task_list:
            normalizer_name, modifying = task_list.popitem(last = False)
            # replace modifying with all observables if keyword "all" is supplied
            if "all" in modifying:
                modifying = observables
                args["using_all_observables"] = True
            else:
                args["using_all_observables"] = False
            logger.debug("Applying normalizer " + normalizer_name)
            normalizer = (self.normalizer_map[normalizer_name])()

            # create a mask for the next operation
            mask = self._mask(observables, modifying)

            # call preprocessor functions, passing any additional args as is
            # modify args here to add in additional arguments passed to preprocessor function
            arr_gen = normalizer.single(arr_gen, mask, observables, **args)
            arr_data, arr_sim = normalizer.paired(arr_data, arr_sim, mask, observables, **args)
            gc.collect()
        
        return arr_data, arr_sim, arr_gen
    
    def preprocess_weight(self, feature_array, weights, observables, **args):
        """
        apply the list of weight preprocessors in sequence and return preprocessed weight

        arguments
        ---------
        feature_array: 2d numpy array 
            feature array with shape (number of events, number of observables in each event)
        weights: 1d numpy array
            event weights corresponding to the feature array
        observables: list of str
            name of observables, for example, "th_pt". weight preprocessing happens after feature preprocessing, so translated observable name is expected instead of branch names
        """

        for function in self.weight_preprocessing_functions:
            weights = function(feature_array, weights, observables, **args)

        return weights

    """
    preprocessor function for weights
    they should have the form
    arguments
    ---------
    feature_array: 2d numpy array 
        feature array with shape (number of events, number of observables in each event)
    weights: 1d numpy array
        event weights corresponding to the feature array
    observables: list of str
        name of observables, for example, "th_pt". weight preprocessing happens after feature preprocessing, so translated observable name is expected instead of branch names
    returns
    -------
    weights: 1d numpy array
        reweighed weight array
    """

    def standardize_weight(self, feature_array, weights, observables, **args):
        mean = np.mean(weights)
        std = np.std(weights)
        weights = (weights - mean) / std
        return weights



# Normalizer classes
# they are implemented in classes for provide better readability as each will have a single and paired normalization function

class Normalizer():
    """
    a base class for normalizers. Other normalizer should inherit from this class and build on these functions.
    """
    def single(self, feature_array, mask, observables, **args):
        """
        single methods should preprocess a single 2d feature array and return the normalized array

        arguments
        ---------
        feature_array: numpy 2d array
            feature array of the shape (number of events, number of observables)
        mask: numpy array
            a mask indicating which observables will be normalized
        observables: numpy array
            indicating the order of observables in the feature array

        returns
        -------
        feature_array: numpy 2d array
            normalized feature array. The observable order should not change.
        """
        return feature_array
    
    def paired(self, feature_array_1, feature_array_2, mask, observables, **args):
        """
        paired method should preprocess 2 feature arrays together, usually sim and gen. They should be normalized using the same
        values (such as average and std) calculated from the combination of both arrays.

        the two feature arrays should have the same number and order of observables (which should be guarenteed if same set of preprocessors are used)

        arguments
        ---------
        feature_array_1: numpy 2d array
            feature array of the shape (number of events, number of observables)
        feature_array_2: numpy 2d array
            a second feature array of the shape (number of events, number of observables)
        mask: numpy array
            a mask indicating which observables will be normalized
        observables: numpy array
            indicating the order of observables in the feature array

        returns
        -------
        feature_array_1: numpy 2d array
            normalized feature array. The observable order should not change.
        feature_array_2: numpy 2d array
            normalized feature array. The observable order should not change.
        """
        return feature_array_1, feature_array_2

class DivideByMeansMagnitude(Normalizer):
    def single(self, feature_array, mask, observables, **args):
        if "using_all_observables" in args and args["using_all_observables"]: # ensure the key exists then check it's True
            mean = np.mean(np.abs(feature_array), axis=0)
            oom = 10**(np.log10(mean).astype(int))
            feature_array = feature_array / oom
        else:
            feature_array_slice = feature_array[:, mask]
            mean = np.mean(np.abs(feature_array_slice), axis=0)
            oom = 10**(np.log10(mean).astype(int))
            feature_array[:, mask] = feature_array_slice / oom
        return feature_array

    def paired(self, feature_array_1, feature_array_2, mask, observables, **args):
        if "using_all_observables" in args and args["using_all_observables"]:
            mean_1, mean_2 = np.mean(np.abs(feature_array_1), axis=0), np.mean(np.abs(feature_array_2), axis=0)
            oom_1, oom_2 = 10**(np.log10(mean_1).astype(int)), 10**(np.log10(mean_2).astype(int))
            oom = np.maximum(oom_1, oom_2)
            feature_array_1, feature_array_2 = feature_array_1 / oom, feature_array_2 / oom
        else:
            feature_array_1_slice, feature_array_2_slice = feature_array_1[:, mask], feature_array_2[:, mask]
            mean_1, mean_2 = np.mean(np.abs(feature_array_1_slice), axis=0), np.mean(np.abs(feature_array_2_slice), axis=0)
            oom_1, oom_2 = 10**(np.log10(mean_1).astype(int)), 10**(np.log10(mean_2).astype(int))
            oom = np.maximum(oom_1, oom_2)
            feature_array_1[:, mask], feature_array_2[:, mask] = feature_array_1_slice / oom, feature_array_2_slice / oom
        return feature_array_1, feature_array_2

class Standardize(Normalizer):
    def single(self, feature_array, mask, observables, **args):
        # optimize memory usage when all observables are used
        if "using_all_observables" in args and args["using_all_observables"]: # ensure the key exists then check it's True
            mean = np.mean(feature_array, axis=0)
            std = np.std(feature_array, axis=0)
            feature_array = (feature_array - mean) / std
        else:
            feature_array_slice = feature_array[:, mask]
            mean = np.mean(feature_array_slice, axis=0)
            std = np.std(feature_array_slice, axis=0)
            feature_array[:, mask] = (feature_array_slice - mean) / std
        return feature_array

    def paired(self, feature_array_1, feature_array_2, mask, observables, **args):
        if "using_all_observables" in  args and args["using_all_observables"]:
            s = np.sum(feature_array_1, axis=0) + np.sum(feature_array_2, axis=0)
            s2 = np.sum(feature_array_1 * feature_array_1, axis=0) + np.sum(feature_array_2 * feature_array_2, axis=0)
            n = len(feature_array_1) + len(feature_array_2)
            
            mean = s / n
            std = np.sqrt(s2 / n - mean * mean)

            feature_array_1, feature_array_2 = (feature_array_1 - mean) / std, (feature_array_2 - mean) / std
        else:
            feature_array_1_slice, feature_array_2_slice = feature_array_1[:, mask], feature_array_2[:, mask]
            s = np.sum(feature_array_1_slice, axis=0) + np.sum(feature_array_2_slice, axis=0)
            s2 = np.sum(feature_array_1_slice * feature_array_1_slice, axis=0) + np.sum(feature_array_2_slice * feature_array_2_slice, axis=0)
            n = len(feature_array_1_slice) + len(feature_array_2_slice)
            
            mean = s / n
            std = np.sqrt(s2 / n - mean * mean)

            feature_array_1[:, mask], feature_array_2[:, mask] = (feature_array_1_slice - mean) / std, (feature_array_2_slice - mean) / std
        return feature_array_1, feature_array_2

# preprocessor instance

preprocessor = None

def initialize(observable_dict, prep_config_path, default_observable_names=None):
    """
    create a preprocessor instance from given parameters

    arguments
    ---------
    observable_dict: dictionary 
        loaded from observable config
    prep_config_path: str
        path to the preprocessor config file, usually located in configs/preprocessor
    default_observable_names: list of str
        the assumed order of observables in feature arrays prior to preprocessing. The assumption is that no other preprocessing step that changes
        this order took place and all data handling operations (such as handling cuts) respects and preserves the original order.
        There can be a special case such that observables in, for example, data and sim are different. The Promise is that preprocessing steps will be applied to each of them
        and the end result will be the same set of observables (as an example, data contains pt, theta, and eta. sim contains px, py, and pz, let's say we somehow want
        to convert both to pt, cos(theta), cos(eta)). Then the list of observables will need to be expilcitly passed when calling preprocessing functions. Instead of calling
        preprocessor.feature_preprocess(data), it needs to be preprocessor.feature_preprocess(observables_of_data, data))
    """
    global preprocessor
    preprocessor = Preprocessor(observable_dict, prep_config_path, default_observable_names)

def get() -> Preprocessor:
    """
    returns
    -------
    preprocessor: Preprocessor
        the preprocessor instance
    """
    return preprocessor