import copy
import tensorflow as tf
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np

import second_version
import third_version
from examples import prepare_examples_with_views
from make_music import recursive_predic, results_to_midi
from musicLoading import make_midi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Blocks warning messages

if __name__ == "__main__":
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # print(device_lib.list_local_devices())

    input_size = 256

    X, y = prepare_examples_with_views(input_size)
    model, callbackItems = third_version.get_model_to_train(True)


    print('Finished prep, shape ', X.shape)

    print('Number of training itmes: ', len(X))

    # model.load_weights('third.hdf5')

    model.fit(X, y, epochs=1024, batch_size=64, initial_epoch=9, validation_split=.2, callbacks=callbackItems)

    # print(model.predict(np.array([[[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]])))



    # starting_notes = [[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]
    # padded_input = [[0] * 8] * (input_size - len(starting_notes)) + starting_notes
    #
    # for chord in padded_input:
    #     for note in chord:
    #         if note != 0:
    #             note -= 21
    #
    # # final_input = []
    # #
    # # #Add time value to notes
    # # for chord in padded_input:
    # #     curList = []
    # #     for note in chord:
    # #         curList.append([note, 2])
    # #
    # #     final_input.append(curList)
    #
    # # converted = ex.convert_quantized_to_input(padded_input)
    #
    # results = recursive_predic(model, padded_input)
    # print(results)
    #
    # results_to_midi(results)

    session.close()

