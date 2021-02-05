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
    inputs = keras.layers.Input(input_shape)
    hidden_layer_1 = keras.layers.Dense(100, activation='relu')(inputs)
    hidden_layer_2 = keras.layers.Dense(100, activation='relu')(hidden_layer_1)
    hidden_layer_3 = keras.layers.Dense(100, activation='relu')(hidden_layer_2)
    outputs = keras.layers.Dense(nclass, activation='softmax')(hidden_layer_3)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.summary()

    return model
