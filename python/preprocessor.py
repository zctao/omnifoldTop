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
    def __init__(self, config_path) -> None:
        """
        arguments
        ---------
        config_path: str, path to the preprocessor configuration file, usually located in configs/preprocess
        """
        self.config = None
        if config_path:
            with open(config_path, "r") as config_file:
                self.config = json.load(config_file)
    
    def preprocess(self, features, datahandler, valid_only):
        """
        arguments
        ---------
        features: list of str, features to be requested from datahandler
        datahandler: a datahandler object that contains the data to be preprocessed
        valid_only: boolean that will be passed directly to datahandler

        returns
        -------
        preprocessed data
        """

        # build a dictionary mapping feature names to assigned preprocessor functions
        dictionary = {}
        for feature in features:
            dictionary[feature] = []
        
        for prp_feature in self.config:
            dictionary[feature] = [function_map[prp_name] for prp_name in self.config[prp_feature]]

        # apply preprocessing by each observable, in original supplied order
        feature_array = None

        for feature in dictionary:
            result = datahandler.get_arrays(feature, valid_only = valid_only)
            # apply preprocessors sequentially
            for preprocessor_function in dictionary[feature]:
                result = preprocessor_function(result)
            
            if feature_array:
                feature_array = np.concatenate((feature_array, result), axis = 1)
            else:
                feature_array = result

        return feature_array