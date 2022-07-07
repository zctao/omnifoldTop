"""
preprocessor class that is used to apply pre training mapping to data, gen, sim
preprocessor only uses trainig variables, extra observables do not enter training and are not preprocessed
the implementation of this class use the assumption that feature array is constructed in the same order as how the observable argument is ordered
remember to update this class if that assumption is changed
"""
import json
import numpy as np

# preprocessor functions defined here

def angle_to_sin_cos():
    pass

def normalize():
    pass

# map from string options to functions

function_map = {
    "angle_to_sin_cos": angle_to_sin_cos,
    "normalize": normalize
}

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
        
        # convert observable dict to a variable dict, mapping branch name to observable name
        self.observable_name_dict = {}
        for ob_name in observable_dict:
            self.observable_name_dict[observable_dict[ob_name]["branch_det"]] = ob_name
            self.observable_name_dict[observable_dict[ob_name]["branch_mc"]] = ob_name

        # read in config, fill those without assigned preprocessor to an empty list
        with open(prep_config_path, "r") as config_file:
            self.config = json.load(config_file)
        for observable in observables:
            if observable not in self.config:
                self.config[observable] = []
        
        # map observable names to index
        self.index_map = {}
        for idx, ob_name in enumerate(observables):
            self.index_map[ob_name] = idx
    
    def _map_branch_names(self, features):
        """
        transform branch names (features) to a map of index to functions

        arguments
        ---------
        featuers: list of str, a list of branch names in the root files

        returns
        -------
        a dictionary from index to functions
        """
        i_to_f = {}
        for feature in features:
            ob_name = self.observable_name_dict[feature]
            idx = self.index_map[ob_name]
            function_name_list = self.config[ob_name]
            i_to_f[idx] = [function_map[function_name] for function_name in function_name_list]
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
        preprocessed feature array, which is returned as an independent copy
        """
        i_to_f = self._map_branch_names(features)
        result = None
        for i in feature_array.shape[1]:
            feature_result = feature_array[:, i]

            # apply preprocessor function is exsits
            if i in i_to_f:
                for function in i_to_f[i]:
                    feature_result = function(feature_result, args)
            # convert feature_result to a 2d array with shape (number of events, 1) if it is a 1d array
            if len(feature_result.shape) < 2: result = np.reshape(result, (*result.shape, 1))

            # append column(s) to result
            if result is not None:
                result = np.concatenate((result, feature_result), axis = 1)
            else:
                result = feature_result
        return result



