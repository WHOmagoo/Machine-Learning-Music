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
        lstm = Bidirectional(CuDNNLSTM(lstmNodes, unit_forget_bias=True), merge_mode='concat')(inputs)
    else:
        lstm = Bidirectional(LSTM(lstmNodes, unit_forget_bias=True), merge_mode='concat')(inputs)

    pred1 = Dense(88, activation='sigmoid')(lstm)

    model = Model(inputs=inputs, outputs=[pred1])
    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath='second.hdf5', verbose=0, save_best_only=True)

    print_summary(model)
    return model, checkpointer