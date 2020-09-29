import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import models

import examples
from examples import remove_rhythm


def recursive_predic(model, startingData, prediciton_count=(16*32)):
    result = np.concatenate([startingData, np.zeros((prediciton_count, examples.num_notes))])
    result = result.reshape((1, result.shape[0], result.shape[1]))

    windowSize = 256

    for i in range(prediciton_count):
        curData = result[:,i:(i + windowSize), :]
        newNotes = model.predict(curData)[0]
        best_note = 0
        best_prob = 0

        second_best_note = 0
        second_prob = 0

        for note_index in range(len(newNotes)):
            if newNotes[note_index] > best_prob:
                second_best_note = best_note
                second_prob = best_prob
                best_prob = newNotes[note_index]
                best_note = note_index
            elif newNotes[note_index] > second_prob:
                second_prob = newNotes[note_index]
                second_best_note = note_index

        result[0, i + windowSize, best_note] = 1
        result[0, i + windowSize, second_best_note] = 1

    print("Raw results", result)

    return result[:,len(startingData):,:]

def results_to_midi(results):
    quantized = []

    for beat in results:
        curBeatNotes = []

        for note_num in range(1, len(beat)):
            if beat[note_num] == 1:
                curBeatNotes.append([note_num + 21 + examples.num_notes // 2, 1])

        if len(curBeatNotes) > 0:
            quantized.append(curBeatNotes)

    midi = examples.make_midi(quantized, 480)
    midi.open("out.mid", 'wb')
    midi.write()

if __name__ == '__main__':
    # print(device_lib.list_local_devices())

    input_size = 256


    # inputs = Input(shape=(256,8))
    # lstm = Bidirectional(LSTM(124), merge_mode='concat')(inputs)
    # pred1 = Dense(88, activation='sigmoid')(lstm)
    #
    # pred = Dropout(.4)(pred1)
    # model = Model(inputs=inputs, outputs=[pred])
    # opt = Adam(lr=1e-3, decay=1e-5)
    #
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.load_weights('best_weights.hdf5')


    midData = examples.load_data("../Music/kunstderfuge.com/scarlatti 24.mid")




    notes = remove_rhythm(midData)
    notes = examples.convert_to_multihot(notes, examples.num_notes, 30)
    notes = notes[:input_size]

    model = models.Sequential()
    #The LSTM needs data with the format of [samples, time steps and features]
    # model.add(layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(169, activation='relu'))
    # model.add(layers.Dense(examples.num_notes, activation='sigmoid'))

    model.add(layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=False, time_major=False))
    model.add(layers.Dense(65, activation='sigmoid'))


    model.load_weights("/home/whomagoo/IdeaProjects/Machine-Learning-Music/Models/Simple_2020-09-28 17:22:10.730312.index")
    # model.load_weights()

    # starting_notes = [[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]
    # padded_input = [[0] * 8] * (input_size - len(starting_notes)) + starting_notes

    results = recursive_predic(model, notes)

    print(results)

    results_to_midi(results[0])