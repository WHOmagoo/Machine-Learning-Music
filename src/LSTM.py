import copy

import tensorflow as tf
from keras import metrics
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Input, Bidirectional
import os
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
import numpy as np

from tensorflow.python.keras.layers import LSTM, Dense

import examples as ex
from examples import prepare_examples, prepare_examples_with_views
from musicLoading import make_midi

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
    result = copy.deepcopy(startingData)

    curData = copy.deepcopy(startingData)
    curData = np.array([curData])
    for i in range(16 * 32):
        newNotes = model.predict(curData)
        notes = []

        for note in range(len(newNotes)):
            if newNotes[i] > .8:
                notes.append(i)

        result.append(notes)
        curData = np.delete(curData[0], 0, axis=0)
        curData = np.append(curData[0], newNotes, axis=0)
        # curData = np.array([curData)

    print("Raw results", result)

    return result

if __name__ == "__main__":
    session = tf.Session()

    input_size = 256

    X, y = prepare_examples_with_views(input_size)

    print('Finished prep, shape ', X[0].shape)

    print('Number of training itmes: ', len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
    # y_train = to_categorical(y_train[:,:1], num_classes=92, dtype='int')
    # y_test = to_categorical(y_test[:,:1], num_classes=92, dtype='int')

    inputs = Input(shape=(256,8), dtype=tf.int64)
    # inputs = Input(shape=X_train[0].shape)

    inputs = Input(shape=X[0].shape)

    lstm = Bidirectional(LSTM(124), merge_mode='concat', dtype=tf.int64)(inputs)
    pred1 = Dense(88, activation='sigmoid')(lstm)

    pred = Dropout(.4)(pred1)


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

    model = Model(inputs=inputs, outputs=pred1)
    
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.input_shape)
    print(model.output_shape)

    model.fit(
        X,
        y, epochs=200, validation_data=(X, y)
        )

    # print(model.predict(np.array([[[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]])))



    starting_notes = [[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]
    padded_input = [[0] * 8] * (input_size - len(starting_notes)) + starting_notes

    # final_input = []
    #
    # #Add time value to notes
    # for chord in padded_input:
    #     curList = []
    #     for note in chord:
    #         curList.append([note, 2])
    #
    #     final_input.append(curList)

    # converted = ex.convert_quantized_to_input(padded_input)

    results = recursive_predic(model, padded_input)

    print(results)

    results_to_midi(results)

    session.close()

