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

