"""
a class for preprocessing observables
"""

import json
import numpy as np
from os.path import join

# if nothing is specified, no preprocessor will be applied
CONFIG_PATH = join("preprocess", "none.json")

PREPROCESSOR = None

def preprocessor(config_path = None):
    """
    returns a single instance of Preprocessor, if it is uninitialized, then initialize it from config_path
    """
    global PREPROCESSOR
    if PREPROCESSOR is None: PREPROCESSOR = Preprocessor(config_path)
    return PREPROCESSOR

# define preprocessing functions here
def angle_to_sin_cos(angle):
    """
    arguments
    ---------
    angle: numpy array (or arraylike object) of angles in radians with shape (number of events,)

    returns
    -------
    numpy array of dimension (number of events, 2)
    """
    return np.stack((np.sin(angle), np.cos(angle)), axis=1)

# function map
function_map = {
    "angle_to_sin_cos" : angle_to_sin_cos
}

class Preprocessor():
    def __init__(self, config_path, observable_dict) -> None:
        """
        arguments
        ---------
        config_path: str, path to the preprocessor configuration file, usually located in configs/preprocess
        observable_dict: observable dictionary read from observable config
        """
        self.config = None
        if config_path:
            with open(config_path, "r") as config_file:
                self.config = json.load(config_file)

        # switch the observable dictionary around, so we can look up a variable name and check if it need to be preprocessed
        # this is possible since physically a variable name can only correspond to one observable
        self.variable_dict = {}
        for observable in observable_dict:
            self.variable_dict[observable_dict[observable]['branch_det']] = observable
            self.variable_dict[observable_dict[observable]['branch_mc']] = observable

    def _preprocess_single(self, features, datahandler, valid_only):
        """
        arguments
        ---------
        features: list of str, features to be requested from datahandler
        datahandler: a datahandler object containing the data
        valid_only: a single boolean

        returns
        -------
        preprocessed data with the shape (number of events, number of features)
        """

        # build a dictionary mapping feature names to assigned preprocessor functions
        dictionary = {}
        for feature in features:
            dictionary[feature] = []
        
        if self.config is not None:
            for prp_feature in self.config:
                dictionary[feature] = [function_map[prp_name] for prp_name in self.config[prp_feature]]

        # apply preprocessing by each observable, in original supplied order
        feature_array = None

        for feature in dictionary:
            result = datahandler.get_arrays(feature, valid_only = valid_only)
            # apply preprocessors sequentially
            for preprocessor_function in dictionary[feature]:
                result = preprocessor_function(result)
            if len(result.shape) < 2: result = np.reshape(result, (*result.shape, 1)) # convert to two dimensional array if not alreay is
            
            if feature_array is not None:
                feature_array = np.concatenate((feature_array, result), axis = 1)
            else:
                feature_array = result

        return feature_array
    
    def preprocess(self, features, datahandlers, valid_only):
        """
        arguments
        ---------
        features: list of list of str, features to be requested from datahandlers
        datahandlers: a list of datahandler objects that contains the data to be preprocessed
        valid_only: a list of boolean that will be passed directly to corresponding datahandlers

        returns
        -------
        n preprocessed data corresponding to n datahandler objects
        """

        # build a dictionary mapping feature names to assigned preprocessor functions
        dictionary = {}
        for feature in features:
            dictionary[feature] = []
        
        if self.config is not None:
            for prp_feature in self.config:
                dictionary[feature] = [function_map[prp_name] for prp_name in self.config[prp_feature]]

        # apply preprocessing by each observable, in original supplied order
        feature_array = None

        for feature in dictionary:
            result = datahandler.get_arrays(feature, valid_only = valid_only)
            # apply preprocessors sequentially
            for preprocessor_function in dictionary[feature]:
                result = preprocessor_function(result)
            if len(result.shape) < 2: result = np.reshape(result, (*result.shape, 1)) # convert to two dimensional array if not alreay is
            
            if feature_array is not None:
                feature_array = np.concatenate((feature_array, result), axis = 1)
            else:
                feature_array = result

        return feature_array