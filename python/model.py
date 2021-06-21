import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def get_callbacks(model_filepath=None):

    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, restore_best_weights=True
    )

    if model_filepath:
        #checkpoint_fp = model_filepath + '_Epoch-{epoch}'
        checkpoint_fp = model_filepath
        CheckPoint = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_fp, verbose=1, monitor='val_loss',
            save_best_only=True, save_weights_only=True
        )

        logger_fp = model_filepath+'_history.csv'
        CSVLogger = keras.callbacks.CSVLogger(
            filename=logger_fp, append=False
        )

        return [CheckPoint, CSVLogger, EarlyStopping]
    else:
        return [EarlyStopping]

def weighted_binary_crossentropy(y_true, y_pred):
    # https://github.com/bnachman/ATLASOmniFold/blob/master/GaussianToyExample.ipynb
    # event weights are zipped with the labels in y_true
    event_weights = tf.gather(y_true, [1], axis=1)
    y_true = tf.gather(y_true, [0], axis=1)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.-epsilon)
    loss = -event_weights * ((y_true) * K.log(y_pred) + (1-y_true) * K.log(1-y_pred))
    return K.mean(loss)

def weighted_categorical_crossentropy(y_true, y_pred):
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

def get_model(input_shape, model_name='dense_3hl', nclass=2):

    model = eval(model_name+"(input_shape, nclass)")

    model.compile(loss=weighted_categorical_crossentropy,
                  #loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.summary()

    return model

def dense_3hl(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hidden_layer_1 = keras.layers.Dense(100, activation='relu')(inputs)
    hidden_layer_2 = keras.layers.Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
    outputs = keras.layers.Dense(nclass, activation='softmax')(hidden_layer_3)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn

def dense_6hl(input_shape, nclass=2):
    inputs = keras.layers.Input(input_shape)
    hidden_layer_1 = keras.layers.Dense(100, activation='relu')(inputs)
    hidden_layer_2 = keras.layers.Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
    hidden_layer_4 = keras.layers.Dense(100, activation='relu')(hidden_layer_3)
    hidden_layer_5 = keras.layers.Dense(100, activation='relu')(hidden_layer_4)
    hidden_layer_6 = keras.layers.Dense(100, activation='relu')(hidden_layer_5)
    outputs = keras.layers.Dense(nclass, activation='softmax')(hidden_layer_6)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn

def pfn(input_shape, nclass=2, nlatent=8):
    # https://arxiv.org/pdf/1810.05165.pdf

    # expected input_shape: (n_particles, n_features...)
    assert(len(input_shape) > 1)
    nparticles = input_shape[0]

    inputs = keras.layers.Input(input_shape, name='Input')

    latent_layers = []
    for i in range(nparticles):
        particle_input = keras.layers.Lambda(lambda x: x[:,i,:], name='Lambda_{}'.format(i))(inputs)

        # per particle map to the latent space
        Phi_1 = keras.layers.Dense(100, activation='relu', name='Phi_{}_1'.format(i))(particle_input)
        Phi_2 = keras.layers.Dense(100, activation='relu', name='Phi_{}_2'.format(i))(Phi_1)
        Phi = keras.layers.Dense(nlatent, activation='relu', name='Phi_{}'.format(i))(Phi_2)
        latent_layers.append(Phi)

    # add the latent representation
    added = keras.layers.Add()(latent_layers)

    F_1 = keras.layers.Dense(100, activation='relu', name='F_1')(added)
    F_2 = keras.layers.Dense(100, activation='relu', name='F_2')(F_1)
    F_3 = keras.layers.Dense(100, activation='relu', name='F_3')(F_2)

    # output
    outputs = keras.layers.Dense(nclass, activation='softmax', name='Output')(F_3)

    nn = keras.models.Model(inputs=inputs, outputs=outputs)

    return nn
