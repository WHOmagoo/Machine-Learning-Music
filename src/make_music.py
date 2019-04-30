import copy
import numpy as np
import tensorflow as tf

import second_version
from examples import remove_rhythm
from musicLoading import make_midi, load_data


def recursive_predic(model, startingData):
    result = copy.deepcopy(startingData)

    result.append([85,84,83,82,81,80,0,0])

    curData = copy.deepcopy(startingData)
    for i in range(16 * 32):
        curData = np.array([curData])
        newNotes = model.predict(curData)[0]
        note_probabilities = [0] * 8
        outputted_notes = [0] * 8

        noteCount = 0

        for note_index in range(len(newNotes)):
            if newNotes[note_index] > note_probabilities[7]:
                for index in range(8):
                    if note_probabilities[index] < newNotes[note_index]:
                        note_probabilities.insert(index, newNotes[note_index])
                        note_probabilities.pop()

                        outputted_notes.insert(index, note_index)
                        outputted_notes.pop()
                        break

        for index in range(7):
            if note_probabilities[index] * .9 > note_probabilities[index + 1]:
                for index2 in range(index, 8):
                    outputted_notes[index2] = 0

        result.append(outputted_notes)
        curData = np.delete(curData[0], 0, axis=0)
        curData = np.append(curData, [note_probabilities], axis=0)
        # curData = np.array([curData)

    print("Raw results", result)

    return result

def results_to_midi(results):
    quantized = []

    for beat in results:
        curBeatNotes = []

        for note in beat:
            if note == 0:
                break;

            curBeatNotes.append([note + 21, 1])

        quantized.append(curBeatNotes)

    midi = make_midi(quantized, 480)
    midi.open("out.mid", 'wb')
    midi.write()

if __name__ == '__main__':
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
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


    midData = load_data("/home/whomagoo/github/MLMusic/Music/kunstderfuge.com/scarlatti 34.mid")

    notes = remove_rhythm(midData)
    notes = notes[:input_size]

    # starting_notes = [[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]
    # padded_input = [[0] * 8] * (input_size - len(starting_notes)) + starting_notes

    i = 0
    for chord in notes:
        j = 0
        for note in chord:
            if note != 0:
                notes[i][j] -= 21

            j+=1
        i+=1

    model, nothing = second_version.get_model_to_train()
    model.load_weights('second.hdf5')

    results = recursive_predic(model, notes)

    print(results)

    results_to_midi(results)

    session.close()