import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Input, Bidirectional
import os
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
import numpy as np

from tensorflow.python.keras.layers import LSTM, Dense

from examples import prepare_examples
from main import make_midi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Blocks warning messages

def get_likliest_two_notes(notesList):
    best = 0
    secondBest = 0
    result = (0,0)
    i = 0
    for note in notesList[0]:
        if note > best:
            result = (i, result[0])
            secondBest = best
            best = note
        i+=1

    finalResult = [0, 0]

    if best > .7:
        finalResult[0] = result[0]

    if secondBest > .8:
        finalResult[1] = result[1]

    return finalResult

def results_to_midi(results):
    quantized = []

    for beat in results:
        curBeatNotes = []

        for note in beat:
            if note == 0:
                break;

            curBeatNotes.append([note, 1])

        quantized.append(curBeatNotes)

    midi = make_midi(quantized, 480)
    midi.open("out.mid", 'wb')
    midi.write()



def recursive_predic(model, startingData):
    result = copy.deepcopy(startingData[0].tolist())

    curData = copy.deepcopy(startingData)
    for i in range(16 * 32):
        newNotes = model.predict(curData)
        notes = get_likliest_two_notes(newNotes)
        notes = [notes[0], notes[1], 0,0,0,0,0,0];

        result.append(notes)
        curData = np.delete(curData[0], 0, axis=0)
        curData = np.append(curData, [notes], axis=0)
        curData = np.array([curData])

    print("Raw results", results)

    return result


if __name__ == "__main__":
    session = tf.Session()

    print('Preparing Examples...')
    X, y = prepare_examples()

    print('X_train:', X)
    print('y_train:', y)
    print(y[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
    y_train = to_categorical(y_train[:,:1], num_classes=92, dtype='int')
    y_test = to_categorical(y_test[:,:1], num_classes=92, dtype='int')


    print('AAAAAAAAAAAAA', y_train[0])
    

    
    print('Shape:', X_train.shape[1:])
    print('Shape', X_train.shape)

    inputs = Input(shape=X_train.shape[1:])

    lstm = Bidirectional(LSTM(124), merge_mode='concat')(inputs)
    pred1 = Dense(92, activation='softmax')(lstm)


    """x = (LSTM(128, input_shape = X_train.shape[1:], activation='relu', return_sequences=True, dtype=tf.int32))(inputs)
    x = (LSTM(128, activation='relu'))(x)
    x = (Dense(32, activation='relu'))(x)
    pred1 = (Dense(1, activation='softmax'))(x)
    pred2 = (Dense(1, activation='softmax'))(x)
    pred3 = (Dense(1, activation='softmax'))(x)
    pred4 = (Dense(1, activation='softmax'))(x)
    pred5 = (Dense(1, activation='softmax'))(x)
    pred6 = (Dense(1, activation='softmax'))(x)
    pred7 = (Dense(1, activation='softmax'))(x)
    pred8 = (Dense(1, activation='softmax'))(x)"""
    model = Model(inputs = inputs, outputs=[pred1])
    
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(
        X_train, 
        y_train, epochs=260, validation_data=
        (X_test, 
        y_test)
        )

    results = recursive_predic(model, np.array([[[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]]))

    print(results)

    results_to_midi(results)



    session.close()

