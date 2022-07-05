"""
a class for preprocessing observables
"""

import json
import numpy as np
from os import join

# if nothing is specified, no preprocessor will be applied
CONFIG_PATH = join("preprocess", "none.json")

PREPROCESSOR = None

def preprocessor(config_path = None):
    """
    returns a single instance of Preprocessor, if it is uninitialized, then initialize it from config_path
    """
    if not PREPROCESSOR: PREPROCESSOR = Preprocessor(config_path)
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
    def _config_to_dict(self, config):
        """
        converts a config dictionary with key value pairs of type (str:list) to (preprocessor function : list) and save in this class

        arguments
        ---------
        config: json object of config

        raises
        ------
        KeyError: if the requested preprocessor is not found
        """
        self.dictionary = {}
        for name in config:
            self.dictionary[function_map[name]] = config[name]
        
    def __init__(self, config_path) -> None:
        """
        arguments
        ---------
        config_path: str, path to the preprocessor configuration file, usually located in configs/preprocess
        """
        config = None
        if config_path:
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
        self._config_to_dict(config)
        
        self.preprocessing_features = set()
        for name in config:
            self.preprocessing_features.update(config[name])
    
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

        base = [feature for feature in features if feature not in self.preprocess_features]
        feature_array = datahandler.get_arrays(base, valid_only = valid_only)

        for preprocess in self.dictionary:
            result = preprocess(datahandler.get_arrays(self.dictionary[preprocess], valid_only = valid_only))
            feature_array = np.concatenate((feature_array, result), axis = 1) # concatenate along the axis representing event features, thus the shape will be (number of events, number of features)

        return feature_array