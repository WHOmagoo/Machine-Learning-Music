import copy
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional, Dense, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.utils import print_summary

from examples import remove_rhythm
from musicLoading import make_midi, load_data

def get_model_to_train(gpu=False):
    inputs = Input(shape=(256,8))
    # inputs = Input(shape=X_train[0].shape)

    lstm = None

    lstmNodes = 88
    if gpu:
        lstm = CuDNNLSTM(256, unit_forget_bias=True, return_sequences=True)(inputs)
        lstm = CuDNNLSTM(256, unit_forget_bias=True, return_sequences=True)(lstm)
        lstm = CuDNNLSTM(lstmNodes, unit_forget_bias=True)(lstm)
    else:
        lstm = LSTM(256, unit_forget_bias=True, return_sequences=True)(inputs)
        lstm = LSTM(256, unit_forget_bias=True, return_sequences=True)(lstm)
        lstm = LSTM(lstmNodes, unit_forget_bias=True)(lstm)

    pred1 = Dense(88, activation='softmax')(lstm)



    model = Model(inputs=inputs, outputs=[pred1])
    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy', 'accuracy'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath='fourth.hdf5', verbose=1, save_best_only=False)

    print_summary(model)
    return model, [checkpointer]