"""
this file is necessary to be used by (currently) both callbacks and modelUtil
it ensures the same naming convention is used
and it avoid error from circular import
"""

def _layer_name(parallel_model_idx, layer_type, dense_depth = 0):
    """
    give proper name to a layer, necessary to follow name conventions since they can be used by other moduels to identify models and layers

    arguments
    ---------
    parallel_model_idx: int
        unique index from 0 to n_models_in_parallel for each parallel model
    layer_type: str
        type of the layer, currently only one of "input", "output", and "dense"
    dense_depth: int
        indicating this layer is the dense_depth th dense layer

    returns
    -------
    name: str
        proper name for the layer following the same naming convention
    """
    if layer_type == "input" or layer_type == "output":
        return "model_{0}_{1}".format(parallel_model_idx, layer_type)
    elif layer_type == "dense":
        return "model_{0}_{1}_{2}".format(parallel_model_idx, layer_type, dense_depth)