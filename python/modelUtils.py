"""
Define model architectures.
"""
# FIXME: Fix support for other types of networks, currently only default dense network works

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from lrscheduler import get_lr_scheduler
from callbacks import EarlyLocking

import plotter

n_models_in_parallel = 2

import logging
logger = logging.getLogger('model')

def get_callbacks(model_filepath=None):
    """
    Set up a list of standard callbacks used while training the models.

    Parameters
    ----------
    model_filepath : str, optional
        If provided, location to save metrics from training the model

    Returns
    -------
    sequence of `tf.keras.callbacks.Callback`
    """
    EarlyLockingCallback = EarlyLocking(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True, n_models_in_parallel=n_models_in_parallel)

    lr_callbacks = get_lr_scheduler().get_callbacks()

    if model_filepath:
        # checkpoint_fp = model_filepath + '_Epoch-{epoch}'
        checkpoint_fp = model_filepath
        CheckPoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_fp,
            verbose=1,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )

        logger_fp = model_filepath + "_history.csv"
        CSVLogger = keras.callbacks.CSVLogger(filename=logger_fp, append=False)
        return [CheckPoint, CSVLogger, EarlyLockingCallback] + lr_callbacks
    else:
        return [EarlyLockingCallback] + lr_callbacks

def weighted_binary_crossentropy(y_true, y_pred):
    """
    Binary crossentropy loss, taking into account event weights.

    Parameters
    ----------
    y_true : (n, 3) tf.Tensor
       Ground truth zipped with event weights.
    y_pred : (n, 2) tf.Tensor
       Predicted categories.

    Returns
    -------
    (n,) tensor
        Calculated loss for each batch
    """
    # https://github.com/bnachman/ATLASOmniFold/blob/master/GaussianToyExample.ipynb
    # event weights are zipped with the labels in y_true
    event_weights = tf.gather(y_true, [1], axis=1)
    y_true = tf.gather(y_true, [0], axis=1)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.-epsilon)
    loss = -event_weights * ((y_true) * K.log(y_pred) + (1-y_true) * K.log(1-y_pred))
    return K.mean(loss)

def weighted_categorical_crossentropy(y_true, y_pred):
    """
    Categorical crossentropy loss, taking into account event weights.

    Parameters
    ----------
    y_true : (n, ncategories + 1) tf.Tensor
        Ground truth zipped with event weights.
    y_pred : (n, ncategories) tf.Tensor
        Predicted cateogires.

    Returns
    -------
    (n,) tf.Tensor
    """
    # event weights are zipped with the labels in y_true
    ncat = y_true.shape[1] - 1
    event_weights = tf.squeeze(tf.gather(y_true, [ncat], axis=1))
    y_true = tf.gather(y_true, list(range(ncat)), axis=1)

    # scale preds so that the class probabilites of each sample sum to 1
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.-epsilon)

    # compute cross entropy
    loss = -event_weights * tf.reduce_sum(y_true * K.log(y_pred), axis=-1)
    return K.mean(loss)

def parse_name_for_dense(model_name):
    """
    Parse the model name and return a list of number of nodes in each layer in 
    case of dense neural network.

    Parameters
    ----------
    model_name : str
        Name of the model to set up. In case if dense network, the expected name
        is "dense_m_n_...", where m, n, ... are number of nodes in each layer 
        or "dense_mxl", where m is the number of nodes in every layer and l is
        the number of layers

    Return
    ------
    A list of positive int
    """
    if 'dense_' in model_name:
        # expected dense_model name: dense_a_b_...
        # where a,a,... are number of nodes in each hidden layer
        # special case: e.g. dense_100x2: two hidden layers each with 100 nodes
        nodes_list = model_name.lstrip('dense_').split('_')
        if len(nodes_list)==1 and 'x' in nodes_list[0]:
            nl = nodes_list[0].split('x')
            assert(len(nl)==2)
            # nl[0]: number of nodes in each layer; nl[1]: number layers
            return [int(nl[0])] * int(nl[1])
        else:
            return [int(n) for n in nodes_list]
    else:
        return []

def get_model(input_shape, nclass=2, model_name='dense_100x3'):
    """
    Build and compile the classifier for OmniFold.

    Parameters
    ----------
    input_shape : sequence of positive int
        Shape of the input layer of the model.
    model_name : str, default: "dense_3hl"
        The name of a function in the `model` module that builds an
        architecture and returns a `tf.keras.models.Model`.
    nclass : positive int, default: 2
        Number of classes in the classifier.

    Returns
    -------
    tf.keras.models.Model
        Model compiled with loss function
        `model.weighted_categorical_crossentropy`, Adam optimizer and
        accuracy metrics.
    """
    # parse model_name
    nodes_list = parse_name_for_dense(model_name)

    if nodes_list:
        model = dense_net(input_shape, nodes_list, nclass)
    else:
        model = eval(model_name+"(input_shape, nclass)")

    optimizer = keras.optimizers.Adam(learning_rate = get_lr_scheduler().get_schedule())

    model.compile(loss=weighted_binary_crossentropy,
                  #loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()

    return model

def train_model(model, X, Y, w, callbacks=[], figname='', batch_size=32768, epochs=100, verbose=1):

    # initalize empty lists
    X_train_list, X_val_list, Yw_train_list, Yw_val_list = [], [], [], []

    # prepare the lists
    for i in range(n_models_in_parallel):
        X_train, X_val, Y_train, Y_val, w_train, w_val = train_test_split(X, Y, w[i])

        # Zip label and weight arrays to use the customized loss function
        Yw_train_list += [np.column_stack((Y_train, w_train))]
        Yw_val_list  += [np.column_stack((Y_val, w_val))]
        X_train_list += [X_train]
        X_val_list += [X_val]

    fitargs = {'callbacks': callbacks, 'epochs': epochs, 'verbose': verbose, 'batch_size': batch_size}

    if n_models_in_parallel == 1:
        model.fit(X_train_list[0], Yw_train_list[0], validation_data=(X_val_list[0], Yw_val_list[0]), **fitargs)
    else:
        model.fit(X_train_list, Yw_train_list, validation_data=(X_val_list, Yw_val_list), **fitargs)

    # FIXME: Y and w are stacked together into Yw and requires separating for plotting
    # if figname:
    #     logger.info(f"Plot model output distributions: {figname}")
    #     preds_train = model.predict(X_train_list, batch_size=batch_size)[:,1]
    #     preds_val = model.predict(X_val_list, batch_size=batch_size)[:,1]
    #     plotter.plot_training_vs_validation(figname, preds_train, Y_train, w_train, preds_val, Y_val, w_val)

def dense_net(input_shape, nnodes=[100, 100, 100], nclass=2):
    """
    A dense neural network classifer. Number of nodes on each layer is specified
    by nnodes

    Parameters
    ----------
    input_shape : sequence of positive int
        Shape of the input layer of the model.
    nnodes : sequence of positive int
        Number of nodes in each hidden layer.
    nclass : positive int, default: 2
        Number of classes in the classifier.

    Returns
    -------
    tf.keras.models.Model
    """
    inputs, outputs = [], []

    for i in range(n_models_in_parallel):
        input_layer = keras.layers.Input(input_shape, name="model_{0}_input".format(i))
        prev_layer = input_layer
        for idx,n in enumerate(nnodes):
            prev_layer = keras.layers.Dense(n, activation="relu", name="model_{0}_dense_{1}".format(i, idx))(prev_layer)

        #output_layer = keras.layers.Dense(nclass, activation="softmax")(prev_layer)
        output_layer = keras.layers.Dense(1, activation="sigmoid", name="model_{0}_output".format(i))(prev_layer)

        inputs += [input_layer]
        outputs += [output_layer]

    return keras.models.Model(inputs=inputs, outputs=outputs)

def pfn(input_shape, nclass=2, nlatent=8):
    """
    A particle flow network [1]_ architecture.

    Parameters
    ----------
    input_shape : sequence of positive int
        Shape of the input layer of the model. Expect at least two
        dimensions: `(n_particles, n_features...)`
    nclass : positive int, default: 2
        Number of classes in the classifier.
    nlatent : positive int, default: 8
        Dimension of the latent space for per-particle representation.

    Returns
    -------
    tf.keras.models.Model

    Notes
    -----
    Particle flow networks learn a mapping from per-particle
    representation to a point in the latent space (of dimension
    `nlatent`), then adds the adds the latent space vectors to get a
    latent event representation. The event representation is the input
    to a learned function that returns the desired output observable.
    Both the particle-to-latent space and the event-to-observable
    steps are learned in the same training loop.

    .. [1] P. T. Komiske et al., "Energy Flow Networks: Deep Sets for
       Particle Jets," arXiv:1810.05165 [hep-ph].

    """
    assert len(input_shape) > 1
    nparticles = input_shape[0]

    inputs = keras.layers.Input(input_shape, name="Input")

    latent_layers = []
    for i in range(nparticles):
        particle_input = keras.layers.Lambda(
            lambda x: x[:, i, :], name="Lambda_{}".format(i)
        )(inputs)

        # per particle map to the latent space
        Phi_1 = keras.layers.Dense(100, activation="relu", name="Phi_{}_1".format(i))(
            particle_input
        )
        Phi_2 = keras.layers.Dense(100, activation="relu", name="Phi_{}_2".format(i))(
            Phi_1
        )
        Phi = keras.layers.Dense(nlatent, activation="relu", name="Phi_{}".format(i))(
            Phi_2
        )
        latent_layers.append(Phi)

    # add the latent representation
    added = keras.layers.Add()(latent_layers)

    F_1 = keras.layers.Dense(100, activation="relu", name="F_1")(added)
    F_2 = keras.layers.Dense(100, activation="relu", name="F_2")(F_1)
    F_3 = keras.layers.Dense(100, activation="relu", name="F_3")(F_2)

    outputs = keras.layers.Dense(nclass, activation="softmax", name="Output")(F_3)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn
