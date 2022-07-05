"""
a class for preprocessing observables
"""

import json
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

class Preprocessor():
    def __init__(self, config_path) -> None:
        """
        arguments
        ---------
        config_path: str, path to the preprocessor configuration file, usually located in configs/preprocess
        """
        if not config_path:
            self.config = None
        else:
            with open(config_path, "r") as config_file:
                self.config = json.load(config_file)