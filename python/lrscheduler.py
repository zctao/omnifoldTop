# This file contains learning schedule functions that is used with tf.keras.callbacks.LearningRateScheduler

import numpy as np
import tensorflow.keras.optimizers.schedules as s

WARM_UP_EPOCHS = 5
PLATEAU = 10

debug = True

import logging
logger = logging.getLogger('lrs')
logger.setLevel(logging.DEBUG)

def debug_learning_rate(lr):
    if debug:
        logger.debug("Current learning rate is {0}".format(float(lr)))

def constant(epoch, lr):
    """
    constant learning rate, independent of epoch
    """
    debug_learning_rate(lr)
    return lr

def warm_up_constant(epoch, lr):
    """
    perform warm up in training with a linearly increasing learning rate for the first WARM_UP_EPOCHS epoch
    """
    debug_learning_rate(lr)
    if epoch == 0:
        return lr / WARM_UP_EPOCHS
    elif epoch < WARM_UP_EPOCHS:
        return (epoch + 1) * lr / epoch
    else:
        return lr

def exponential_decay(epoch, lr):
    """
    exponential decay of learning rate
    """
    debug_learning_rate(lr)
    return lr * np.exp(-0.3)

def warm_up_plateau_decay(epoch, lr):
    """
    perform warm up with a linearly increasing learning rate, hold for PLATEAU epochs, then start exponential decay
    """
    debug_learning_rate(lr)
    if epoch < WARM_UP_EPOCHS + PLATEAU:
        return warm_up_constant(epoch, lr)
    else:
        return exponential_decay(epoch, lr)

"""
put functions and their names as dictionary here to use with run params
"""
scheduler_dict = {
    "constant" : constant,
    "warmc" : warm_up_constant,
    "exp" : exponential_decay,
    "warmexp" : warm_up_plateau_decay
}

def get_scheduler(name):
    """
    returns the requested learning rate scheduler, returns None if the requested scheduler does not exist
    """
    if name in scheduler_dict:
        return scheduler_dict[name]
    else:
        return None