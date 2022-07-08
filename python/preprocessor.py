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

# preprocessor instance

preprocessor = None

def initialize(observables, observable_dict, prep_config_path):
    """
    create a preprocessor instance from given parameters

    arguments
    ---------
    observables: list of str, name of the observables used in training
    observable_dict: dictionary loaded from observable config
    prep_config_path: str, path to the preprocessor config file, usually located in configs/preprocessor
    """
    global preprocessor
    preprocessor = Preprocessor(observables, observable_dict, prep_config_path)

def get():
    """
    returns
    -------
    the preprocessor instance
    """
    return preprocessor

# preprocessor class

class Preprocessor():
    def __init__(self, observables, observable_dict, prep_config_path) -> None:
        """
        arguments
        ---------
        observables: list of str, name of the observables used in training
        observable_dict: dictionary loaded from observable config
        prep_config_path: str, path to the preprocessor config file, usually located in configs/preprocessor
        """

        # map from string options to functions

        self.function_map = {
            "angle_to_sin_cos": self._angle_to_sin_cos,
            "normalize": self._normalize
        }

        # convert observable dict to a variable dict, mapping branch name to observable name
        self.observable_name_dict = {}
        for ob_name in observable_dict:
            self.observable_name_dict[observable_dict[ob_name]["branch_det"]] = ob_name
            self.observable_name_dict[observable_dict[ob_name]["branch_mc"]] = ob_name

        # read in config, fill those without assigned preprocessor to an empty list
        with open(prep_config_path, "r") as config_file:
            self.config = json.load(config_file, object_pairs_hook=OrderedDict)

        # dictionary for normalization function to ensure normalized result respects relative magnitudes in original dataset
        self.normalization_dictionary = {}

    # preprocessor functions defined here

    def _angle_to_sin_cos(self, feature_array, **args):
        """
        maps an angle to its sine and cosine

        arguments
        ---------
        feature_array: 1d numpy array representing of some angle observable

        returns
        -------
        a 2d numpy array of the same length of [sine, cosine]
        """
        return np.stack((np.sin(feature_array), np.cos(feature_array)))

    def _normalize(self, feature_array, **args):
        """
        normalize given feature array to a mean of 0 and standard deviation of 1

        arguments
        ---------
        feature_array: 1d numpy array
        pairing: str, an indicator for pairing feature arrays into groups
        idx: int, column index of this slice of feature array, representing which observable it represents
        """
        pairing = args['pairing']
        idx = args['idx']
        if pairing not in self.normalization_dictionary:
            self.normalization_dictionary[pairing] = []
        if idx not in self.normalization_dictionary[pairing]:
            divisor = 

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
        observables = [self.observable_name_dict[feature] for feature in features]

        # use as a checklist to mark the items that are done
        task_list = self.config.copy()

        
        i_to_f = self._map_branch_names(features)
        result = None
        for i in feature_array.shape[1]:
            feature_result = feature_array[:, i]

            # apply preprocessor function if exsits
            if i in i_to_f:
                for function in i_to_f[i]:
                    args['idx'] = i
                    feature_result = function(feature_result, **args)
            # convert feature_result to a 2d array with shape (number of events, 1) if it is a 1d array
            if len(feature_result.shape) < 2: result = np.reshape(result, (*result.shape, 1))

            # append column(s) to result
            if result is not None:
                result = np.concatenate((result, feature_result), axis = 1)
            else:
                result = feature_result
        return result



