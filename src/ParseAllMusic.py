import examples as ex
from music21.midi import MidiFile
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import preprocessing
from tensorflow.keras import metrics
from tensorflow import data
import numpy as np
from tensorflow.keras.preprocessing import sequence

import sys
import datetime

rootDir = "/Machine-Learning-Music/"
inputPath = rootDir + "Models/Music3_2020-09-26 13:24:04.645036.index"
outputPath = rootDir + "Models/GOOF_" + str(datetime.datetime.now())




if __name__ == '__main__':
    print("Starting")
    # midi = MidiFile()
    # midi.open("/home/whomagoo/IdeaProjects/Machine-Learning-Music/Music/kunstderfuge.com/handel 577.mid", 'rb')
    # midi.read()
    # midi.open("/home/whomagoo/IdeaProjects/Machine-Learning-Music/Music/tmp/handel 577.mid", 'wb')
    # midi.write()

    if(len(sys.argv) > 1):
        rootDir = sys.argv[1]
        print("****\nNew Root Dir")
        print(rootDir)

    all_songs = ex.prepare_examples_for_series_generator(rootDir)
    number_of_outputs = all_songs[0].shape[1]

    window_size = 256

    batchSize = 64

    # data_generator = sequence.TimeseriesGenerator(x, y, length=256, batch_size=batchSize)

    input_data = None

    windows = [0] * len(all_songs)

    input_data = None

    for index, song in all_songs.items():
        #Dictionary.items() gives items wrapped in a tuple of (key, value)
        x = song[:-window_size, :]
        y = song[window_size:, :]
        window = preprocessing.timeseries_dataset_from_array(data=x, targets=y, sequence_length=window_size, sequence_stride=1, sampling_rate=1, shuffle=True, batch_size=batchSize)
        if input_data is None:
            input_data = window
        else:
            input_data = input_data.concatenate(window)

    data_point_count = 0

    # data_generator_unsliced = np.stack(list(data_generator_unsliced)[0])

    # y = all[:, 1]
    # data_generator_slices = data.Dataset.from_tensor_slices(data_generator_unsliced, y)


    #tensorflow.python.framework.errors_impl.InvalidArgumentError:  Incompatible shapes: [32,149] vs. [32,65]
    model = models.Sequential()
    #The LSTM needs data with the format of [samples, time steps and features]
    model.add(layers.LSTM(512, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True, time_major=False))
    model.add(layers.LSTM(256, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True, time_major=False))
    model.add(layers.LSTM(124, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=False, time_major=False))
    model.add(layers.Dense(number_of_outputs, activation='sigmoid'))


    inputPath = None

    if inputPath is not None:
        print("Loading Model From A file")
        print(inputPath)
        model.load_weights(inputPath)

    class_weights = {}

    class_weights[0] = .15
    non_zero_class_weight =  (1 - class_weights[0]) / (number_of_outputs - 1)
    for i in range (1, number_of_outputs):
        class_weights[i] = 1


    sum = 0
    for key, val in class_weights.items():
        sum += val

    print("Weights sum to " + str(sum))

    loss_fn = losses.BinaryCrossentropy()
    opt = optimizers.Adam(learning_rate=.03)

    myMetric = metrics.CategoricalCrossentropy()

    model.compile(optimizer=opt, loss=loss_fn, metrics=[myMetric, metrics.FalseNegatives()])


    cp_callback = callbacks.ModelCheckpoint(filepath=outputPath,
                                            save_weights_only=True,
                                            verbose=1)

    print(outputPath)

    model.fit(input_data,
              epochs=5000,
              callbacks=[cp_callback],
              class_weight=class_weights,
              )

    print("finished")
