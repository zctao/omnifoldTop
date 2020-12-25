from tensorflow import keras
from tensorflow.keras import layers

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

def get_model(input_shape, nclass=2):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(layers.Dense(nclass, activation='softmax', kernel_initializer='he_uniform'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()

    return model
