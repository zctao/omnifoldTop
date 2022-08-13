"""
this file includes custom callback class(es) for tensorflow keras
"""
from tensorflow import keras
from tensorflow import float32
import numpy as np
from layer_namer import _layer_name

import logging
logger = logging.getLogger('callbacks')
logger.setLevel(logging.DEBUG)

class PrintLearningRate(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.model.optimizer._decayed_lr(float32).numpy()
        logger.debug("Current learning rate is {0}".format(learning_rate))

class EarlyLocking(keras.callbacks.Callback):
    """
    Monitors the selected performance parameter for all the parallel models. Once the epoch without improvement for a parallel model exceeds
    set patience, it is locked out from training with trainable set to false. Its weights will be restored to that of patience epoch ago (best result).
    The entire training process will be terminated when every parallel model has stopped improving.
    """
    def __init__(self, monitor, patience, verbose, restore_best_weights, n_models_in_parallel):
        """
        initalize the early locking callback

        arguments
        ---------
        monitor: str
            a performance monitor such as "val_loss" or "loss"
        patience: int
            the number of epoch to run before stopping training after the monitor has stopped improving
        verbose: int
            verbosity of the callback, it has no effect at the moment
        restore_best_weights: boolean
            whether the best weights for each parallel model will be restored
        n_models_in_parallel: int
            the number of parallel models
        """
        super(EarlyLocking, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.n_models_in_parallel = n_models_in_parallel
    
    def on_train_begin(self, logs=None):
        """
        initialization at the before any epoch starts
        """
        self.best_weights = None
        self.best_monitor_value = np.full(self.n_models_in_parallel, np.Infinity)
        # a count down for each parallel model before it goes to 0
        self.counters = np.full(self.n_models_in_parallel, self.patience)

        # build a name layer dictionary
        self.layer_idx_dict = {}
        for idx, layer in enumerate(self.model.layers):
            self.layer_idx_dict[layer.name] = idx

    def _monitor_key(self, parallel_model_idx):
        """
        get the name of the monitor value for the parallel model, following keras' naming convention

        arguments
        ---------
        parallel_model_idx: int
            the index of the parallel model
        """
        if self.monitor == "val_loss":
            return "val_"+_layer_name(parallel_model_idx, "output")+"_loss"
        else:
            return _layer_name(parallel_model_idx, "output")+"_"+self.monitor

    def on_epoch_end(self, epoch, logs=None):
        self.counters -= 1
        # initialize self.best_weights if this is the first epoch
        if self.best_weights == None:
            # the initialization should not be count towards patience
            self.counters += 1
            logger.debug("loading initial set of weights")
            self.best_weights = {}
            for layer in self.model.layers:
                self.best_weights[layer.name] = layer.get_weights()

        # update counter, best_weights, and best_monitor_value if improvement is seen for each individual models
        for model_idx in range(self.n_models_in_parallel):
            monitor_key = self._monitor_key(model_idx) if self.n_models_in_parallel > 1 else self.monitor
            new_monitor_val = logs[monitor_key]
            # if new best is achieved, and the model is still in training (not due to random fluctuations)
            if new_monitor_val < self.best_monitor_value[model_idx] and self.counters[model_idx] >= 0:
                # if we achieved better result than before
                logger.debug("improvement by {0}, saving best set of weight for parallel model {1}".format(self.best_monitor_value[model_idx] - new_monitor_val, model_idx))
                # update best weight
                for layer in self.model.layers:
                    if "model_{0}".format(model_idx) in layer.name:
                        self.best_weights[layer.name] = layer.get_weights()

                # reset counter
                self.counters[model_idx] = self.patience

                # update best_monitor_value
                self.best_monitor_value[model_idx] = new_monitor_val

        # lock models and restore best weight when its counter goes to 0
        for model_idx in range(self.n_models_in_parallel):
            if self.counters[model_idx] == 0:
                logger.debug("restoring best set of weights for parallel model {0}".format(model_idx))
                # restore best weights and set them to untrainable
                for layer_name in self.best_weights:
                    if "model_{0}".format(model_idx) in layer_name:
                        layer = self.model.layers[self.layer_idx_dict[layer_name]]
                        layer.set_weights(self.best_weights[layer_name])
                        layer.trainable = False

        # early stopping when all counters are less than or equal to 0
        if np.all(self.counters <= 0):
            logger.debug("early stopping")
            self.model.stop_training = True

