import copy
import os

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional, Dense, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import Adam
from keras.utils import print_summary
import examples

def get_model_to_train(gpu=False):
    inputs = Input(shape=(256,examples.notes_per_chord))
    # inputs = Input(shape=X_train[0].shape)

    lstmNodes = examples.num_notes

    if gpu:
        lstm = CuDNNLSTM(256, unit_forget_bias=True, return_sequences=True)(inputs)
        lstm = CuDNNLSTM(128, unit_forget_bias=True, return_sequences=True)(lstm)
        dropout = Dropout(.2)(lstm)
        lstm = CuDNNLSTM(lstmNodes, unit_forget_bias=True)(dropout)
    else:
        lstm = LSTM(256, unit_forget_bias=True, return_sequences=True)(inputs)
        lstm = LSTM(128, unit_forget_bias=True, return_sequences=True)(lstm)
        lstm = Dropout(.2)(lstm)
        lstm = LSTM(lstmNodes, unit_forget_bias=True)(lstm)

    drop = Dropout(.2)(lstm)
    pred1 = Dense(examples.num_notes, activation='softmax')(drop)



    model = Model(inputs=inputs, outputs=[pred1])
    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy', 'accuracy'])

    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto')
    checkpointer = ModelCheckpoint(filepath = os.path.join('checkpoints', 'version4_1_1-e{epoch:03d}-ca{categorical_accuracy:.3f}-vca{val_categorical_accuracy:.3f}.hdf5'), verbose=1, save_best_only=False)

    print_summary(model)
    return model, [checkpointer]
