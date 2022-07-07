"""
preprocessor class that is used to apply pre training mapping to data, gen, sim
preprocessor only uses trainig variables, extra observables do not enter training and are not preprocessed
the implementation of this class use the assumption that feature array is constructed in the same order as how the observable argument is ordered
remember to update this class if that assumption is changed
"""
import json

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