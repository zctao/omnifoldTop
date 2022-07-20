"""
this file includes custom callback class(es) for tensorflow keras
"""
from tensorflow import keras
from tensorflow import float32

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
    def __init__(self, monitor, patience, verbose, restore_best_weights):
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
        """
        super(EarlyLocking, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

    def on_epoch_end(self, epoch, logs=None):
        for key, value in self.model._get_trainable_state().items():
            print(key, value)
            if(key.name == "dense_4") and epoch > 2:
                key.trainable = False

