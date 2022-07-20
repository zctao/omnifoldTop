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
    def __init__(self, )

    def on_epoch_end(self, epoch, logs=None):
        for key, value in self.model._get_trainable_state().items():
            print(key, value)
            if(key.name == "dense_4") and epoch > 2:
                key.trainable = False

