"""
preprocessor class that is used to apply pre training mapping to data, gen, sim
preprocessor only uses trainig variables, extra observables do not enter training and are not preprocessed
the implementation of this class use the assumption that feature array is constructed in the same order as how the observable argument is ordered
remember to update this class if that assumption is changed
the preprocessor config should be written in the format of
{
    name of preprocessor function : [list of observables to apply],
    other preprocessing tasks
}
the preprocessor functions will be run in the same order from top to bottom
"""
import json
import numpy as np
from collections import OrderedDict
import gc

import logging
logger = logging.getLogger('Preprocessor')
logger.setLevel(logging.DEBUG)

# preprocessor class

class Preprocessor():
    def __init__(self, observable_dict, prep_config_path) -> None:
        """
        arguments
        ---------
        observables: list of str, name of the observables used in training
        observable_dict: dictionary loaded from observable config
        prep_config_path: str, path to the preprocessor config file, usually located in configs/preprocessor
        """

        logger.info("Initializing Preprocessor")

        # map from string options to functions

        self.function_map = {
            "angle_to_sin_cos": self._angle_to_sin_cos,
            "angle_to_sin": self._angle_to_sin,
            "angle_to_cos": self._angle_to_cos
        }

        # convert observable dict to a variable dict, mapping branch name to observable name
        self.observable_name_dict = {}
        for ob_name in observable_dict:
            self.observable_name_dict[observable_dict[ob_name]["branch_det"]] = ob_name
            self.observable_name_dict[observable_dict[ob_name]["branch_mc"]] = ob_name

        # read in config, fill those without assigned preprocessor to an empty list
        with open(prep_config_path, "r") as config_file:
            self.config = json.load(config_file, object_pairs_hook=OrderedDict)
        
        # save the input on whether the utility functions should be used for outside callers, defaults to False
        if "preprocess_utility" in self.config:
            self.utility = self.config["preprocess_utility"]
            # should be removed to not confuse the function_map
            del self.config["preprocess_utility"]
        else:
            self.utility = True

        # dictionary for normalization function to ensure normalized result respects relative magnitudes in original dataset
        self.normalization_dictionary = {}

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
        feature_array: 1d numpy array representing of some angle observable
        mask: list of boolean, indicating which observables to be modified
        observables: list of str, observable names indicating their position in the feature array

        returns
        -------
        a 2d numpy array of the shape (number of events, number of observabless) with the original observables replaced by sine and cosine under the same name
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
        feature_array: 1d numpy array representing of some angle observable
        mask: list of boolean, indicating which observables to be modified
        observables: list of str, observable names indicating their position in the feature array

        returns
        -------
        a 2d numpy array of the shape (number of events, number of observabless) with the original observables replaced by sine under the same name
        """

        feature_array[:, mask] = np.sin(feature_array[:, mask])

        return feature_array, observables

    def _angle_to_cos(self, feature_array, mask, observables, **args):
        """
        maps an angle to cosine of that angle

        arguments
        ---------
        feature_array: 1d numpy array representing of some angle observable
        mask: list of boolean, indicating which observables to be modified
        observables: list of str, observable names indicating their position in the feature array

        returns
        -------
        a 2d numpy array of the shape (number of events, number of observabless) with the original observables replaced by cosine under the same name
        """

        feature_array[:, mask] = np.cos(feature_array[:, mask])

        return feature_array, observables


    # other functions
    def _get_index_map(self, observables):
        """
        get the index map by remapping the list of observables to their position in the list

        arguments
        ---------
        observables: observable names, these refer to the physical observables instead of root file branch names
        """
        index_map = {}
        for idx, ob_name in enumerate(observables):
            index_map[ob_name] = idx
        return index_map
    
    def _map_idx_to_function(self, observables):
        """
        transform observable names to a map of index to functions

        arguments
        ---------
        observables: list of str, a list of branch names in the root files

        returns
        -------
        a dictionary from index to functions
        """
        i_to_f = {}
        for idx, ob_name in enumerate(observables):
            function_name_list = self.config[ob_name]
            # add preprocessor functions associated to the 'all' keyword
            if 'all' in self.config: function_name_list.append(self.config['all'])
            i_to_f[idx] = [self.function_map[function_name] for function_name in function_name_list]
        return i_to_f

    def _mask(self, observables, modify):
        """
        return a mask indicating whether each index is to be included in the preprocessing

        arguments
        ---------
        observables: the list of observable names indicating position of the observables in the feature array
        modify: list of str, the name of the observables that will be modified by the next preprocessor

        returns
        -------
        an 1d numpy array of boolean indicating whether each index is to be modified
        """
        mask = np.zeros(np.shape(observables))
        for ob_name in modify:
            mask[observables == ob_name] = 1
        mask = mask == 1
        return mask


    def preprocess(self, features, feature_array, **args):
        """
        preprocess feature_array by applying requested preprocessor functions

        arguments
        ---------
        features: name of root tree branchs
        feature_array: 2d numpy array with shape (number of events, observables in each event)
        args: extra parameters that will be passed directly to supported preprocessor functions

        returns
        -------
        preprocessed feature array
        """
        # convert feature names to observable names
        observables = np.array([self.observable_name_dict[feature] for feature in features])

        # use as a checklist to mark the items that are done
        task_list = self.config.copy()

        logger.debug("Observable order before preprocessing: "+str(observables))

        while task_list:
            function_name, list_of_observable = task_list.popitem(last = False)
            logger.debug("Applying preprocessing function " + function_name)
            function = self.function_map[function_name]

            # create a mask for the next operation
            mask = self._mask(observables, list_of_observable)

            # call preprocessor function, passing any additional args as is
            # modify args here to add in additional arguments passed to preprocessor function
            feature_array, observables = function(feature_array, mask, observables, **args)

            logger.debug("Observable order after preprocessing round: "+str(observables))

        gc.collect()
        
        # return the feature array after preprocessing
        return feature_array

    def use_utility(self):
        """
        returns
        -------
        whether the config specified the utility functions to be used
        """
        return self.utility

    # other functions that can be called directly as an utility

    def standardize(self, feature_array):
        """
        standardize the given feature array

        arguments
        ---------
        feature_arrays: numpy 2d arrays of shape (number of events, features in each event)

        returns
        -------
        feature_array standardized to a mean of 0 and standard deviation of one
        """

        # compute mean over all feature_arrays together
        mean = np.mean(feature_array, axis=0)
        std = np.std(feature_array, axis=0)

        return (feature_array - mean) / std

    def group_standardize(self, feature_arrays):
        """
        standardize the given feature arrays using the same factors to retain the relative comparision between them

        arguments
        ---------
        feature_arrays: tuple of numpy 2d arrays of shape (number of events, features in each event)

        returns
        -------
        tuple of feature_arrays as a group standardized to a mean of 0 and standard deviation of one, but not necessarily for each one of them
        """

        # compute mean over all feature_arrays together
        combine = np.concatenate(feature_arrays, axis=0)
        mean = np.mean(combine, axis=0)
        std = np.std(combine, axis=0)

        feature_arrays = ((feature_array - mean) / std for feature_array in feature_arrays)

        return feature_arrays
        

# preprocessor instance

preprocessor = None

def initialize(observable_dict, prep_config_path):
    """
    create a preprocessor instance from given parameters

    arguments
    ---------
    observables: list of str, name of the observables used in training
    observable_dict: dictionary loaded from observable config
    prep_config_path: str, path to the preprocessor config file, usually located in configs/preprocessor
    """
    global preprocessor
    preprocessor = Preprocessor(observable_dict, prep_config_path)

def get() -> Preprocessor:
    """
    returns
    -------
    the preprocessor instance
    """
    return preprocessor