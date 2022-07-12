# This class serves as a simpler combined interface to apply a learning rate scheduler
# There are two main different ways to support this, through either
# tf.keras.callbacks.LearningRateScheduler
# tf.keras.optimizers.schedules
# The two methods each has some nice features that are not trivially available in the other
# note two methods can not be used together, since using schedules takes up the place for learning rate

import numpy as np
import tensorflow.keras.optimizers.schedules as schedules
import tensorflow.keras.callbacks as callbacks

WARM_UP_EPOCHS = 5
PLATEAU = 10

debug = False

import logging
logger = logging.getLogger('lrs')
logger.setLevel(logging.DEBUG)

lrscheduler = None

class LearningRateScheduler():
    def __init__(self, initial_learning_rate, scheduler_names, schedule_args):
        """
        Arguments
        ---------
        initial_learning_rate : inital learning rate
        scheduler_name : list of names refering to the scheduler / callback to be applied, there can at most be one schedule, but many callbacks
        schedule_args : dictionary, extra arguments for the schedule, required for using "piecewised"

        Raise
        -----
        Exception if more than 1 learning schedule is requested
        Exception if callbacks and schedules are being used together
        """

        self.inital_learning_rate = initial_learning_rate

        if scheduler_names is None : scheduler_names = []
        if schedule_args is None: schedule_args = {}

        self.callback_names = [name for name in scheduler_names if name in scheduler_dict]
        self.schedules_names = [name for name in scheduler_names if name in schedules_dict]
        self.callbacks = []
        self.schedule = None

        # enforce the requirements
        # 1. at most 1 schedule
        assert(len(self.schedules_names) <= 1)
        assert((not self.schedules_names) or (not self.callback_names))

        # assemble callbacks
        for callback_name in self.callback_names:
            self.callbacks += [callbacks.LearningRateScheduler(scheduler_dict[callback_name])]

        # assemble schedules
        for schedule_name in self.schedules_names:
            if schedule_args:
                self.schedule = schedules_dict[schedule_name](**schedule_args)
            # defaults
            elif schedule_name in ["cosined", "cosinedr", "polynomiald"]:
                # initial learning rate, decay steps
                self.schedule = schedules_dict[schedule_name](initial_learning_rate, 5000)
            elif schedule_name in ["expd", "inversetd"]:
                # inital learning rate, decay steps, decay rate
                self.schedule = schedules_dict[schedule_name](initial_learning_rate, 1000, 0.95)
    
    def get_callbacks(self):
        """
        return the callbacks, an empty list if no callback is requested
        """
        return self.callbacks
    
    def get_schedule(self):
        """
        return the schedule, just initial learning rate if no schedule is requested
        """
        return self.schedule if self.schedule is not None else self.inital_learning_rate

def debug_learning_rate(lr):
    if debug:
        logger.debug("Current learning rate is {0}".format(float(lr)))

def constant(epoch, lr):
    """
    constant learning rate, independent of epoch
    """
    # debug_learning_rate(lr)
    return lr

def warm_up_constant(epoch, lr):
    """
    perform warm up in training with a linearly increasing learning rate for the first WARM_UP_EPOCHS epoch
    """
    # debug_learning_rate(lr)
    if epoch == 0:
        return lr / WARM_UP_EPOCHS
    elif epoch < WARM_UP_EPOCHS:
        return (epoch + 1) * lr / epoch
    else:
        return lr


# obsolete, use schedules
# def exponential_decay(epoch, lr):
#     """
#     exponential decay of learning rate
#     """
#     debug_learning_rate(lr)
#     return lr * np.exp(-0.3)

# obsolete, basically warmc + expd schedule
# def warm_up_plateau_decay(epoch, lr):
#     """
#     perform warm up with a linearly increasing learning rate, hold for PLATEAU epochs, then start exponential decay
#     """
#     debug_learning_rate(lr)
#     if epoch < WARM_UP_EPOCHS + PLATEAU:
#         return warm_up_constant(epoch, lr)
#     else:
#         return exponential_decay(epoch, lr)

def init_lr_scheduler(initial_learning_rate, scheduler_names, schedule_args):
    """
    Arguments
    ---------
    initial_learning_rate : inital learning rate
    scheduler_name : list of names refering to the scheduler / callback to be applied, there can at most be one schedule, but many callbacks
    schedule_args : dictionary, extra arguments for the schedule, required for using "piecewised"

    Raise
    -----
    Exeption if more than 1 learning schedule is requested
    """
    global lrscheduler
    lrscheduler = LearningRateScheduler(initial_learning_rate, scheduler_names, schedule_args)

def get_lr_scheduler()->LearningRateScheduler:
    """
    returns the learning rate scheduler
    """
    return lrscheduler

"""
put functions and their names as dictionary here to use with run params
"""
# functions that is to be used with tensorflow.keras.callbacks.LearningRate
scheduler_dict = {
    "constant" : constant,
    "warmc" : warm_up_constant,
}

# giving names to tf.keras.optimizers.schedules
schedules_dict = {
    "cosined" : schedules.CosineDecay,
    "cosinedr" : schedules.CosineDecayRestarts,
    "expd" : schedules.ExponentialDecay,
    "inversetd" : schedules.InverseTimeDecay,
    "piecewised" : schedules.PiecewiseConstantDecay,
    "polynomiald" : schedules.PolynomialDecay
}
