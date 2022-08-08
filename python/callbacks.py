"""
this file includes custom callback class(es) for tensorflow keras
"""
from tensorflow import keras
from tensorflow import float32

import logging
logger = logging.getLogger('callbacks')
logger.setLevel(logging.DEBUG)

class PrintLearningRate(keras.callbacks.Callback):
    """
    a callback for printing out current learning rates at the end of each epochs at debug level
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        print out the current learning rate at debug level

        arguments
        ---------
        epoch: int
            current epoch number (not used)
        logs: tf keras log
            keras training log (not used)
        """
        learning_rate = self.model.optimizer._decayed_lr(float32).numpy()
        logger.debug("Current learning rate is {0}".format(learning_rate))

