from time import sleep

import pandas as pd
from  sklearn import preprocessing
import tensorflow as tf
import datetime
import numpy as np
import os

rootDir = "/Machine-Learning-Music/Music/"
dataPath = rootDir +  "diabetic_data.csv"
outputPath = rootDir + "savedModel" + str(datetime.datetime.now()) + ".ckpt"
inputPath = None

def getData():
    #Need to remove the following
    # weight, payer code, max_glu_serum, A1Cresult
    useColumns = ["race",
                  "gender",
                  "age",
                  "admission_type_id",
                  "discharge_disposition_id",
                  "admission_source_id",
                  "time_in_hospital",
                  "num_lab_procedures",
                  "num_procedures",
                  "num_medications",
                  "diag_1",
                  "diag_2",
                  "diag_3",
                  "number_diagnoses",
                  "readmitted"]
    allData = pd.read_csv(dataPath, usecols=useColumns)

    print("*******")
    print(allData)
    print("*******")

    #One hot encode race
    race_onehot = pd.get_dummies(allData["race"], prefix="val")
    allData = pd.concat([race_onehot, allData], axis=1, sort=False)
    #remove original column for race
    del allData["race"]


    print("Race Shape:")
    print(race_onehot.shape)

    #One hot encode gender
    gender_onehot = pd.get_dummies(allData["gender"], prefix="val")
    allData = pd.concat([gender_onehot,allData], axis=1, sort=False)
    #remove original column for race and extra Unkown/Invalid gneder
    del allData["gender"]
    del allData["val_Unknown/Invalid"]

    #One hot encode admission_type_id
    admission_type_id = pd.get_dummies(allData["admission_type_id"], prefix="val")
    allData = pd.concat([admission_type_id, allData], axis=1, sort=False)
    #remove original column for admission_type_id
    del allData["admission_type_id"]

    #One hot encode discharge_disposition_id
    discharge_disposition_id = pd.get_dummies(allData["discharge_disposition_id"], prefix="val")
    allData = pd.concat([discharge_disposition_id, allData], axis=1, sort=False)
    #remove original column for race
    del allData["discharge_disposition_id"]

    #One hot encode admission_source_id
    admission_source_id = pd.get_dummies(allData["admission_source_id"], prefix="val")
    allData = pd.concat([admission_source_id, allData], axis=1, sort=False)
    #remove original column for race
    del allData["admission_source_id"]

    cleanedAge = allData["age"].str.extract(r'(\d+)-', expand=False)
    allData["age"] = pd.to_numeric(cleanedAge)

    #Clean all the diag numbers
    i = 0
    for i in range(1,4):
        col = "diag_" + str(i)
        cleanedDiag = allData[col].str.extract(r'(\d+)', expand=False).fillna(0)
        allData[col] = pd.to_numeric(cleanedDiag)

    cleanY = allData["readmitted"].replace(to_replace=['NO', r'([^N][^O])+'], value=[0,1], regex=True)
    allData["readmitted"] = cleanY

    y_onehot = pd.get_dummies(allData["readmitted"], prefix="val")
    allData = pd.concat([allData, y_onehot], axis=1, sort=False)
    #remove original column for race and extra Unkown/Invalid gneder
    del allData["readmitted"]

    #Transoform the data using a MinMaxScaler, should change this be different depending on the column
    dataTransformer = preprocessing.MinMaxScaler().fit(allData)
    normalizedData = dataTransformer.transform(allData)

    npData = normalizedData

    print(npData.shape)
    print(npData.dtype)
    print(npData)



    shuffled = tf.random.shuffle(npData)

    trainSize = int(shuffled.shape[0] * .8)

    x = shuffled[:,:-2]
    y = shuffled[:,-2:]


    x_train = x[:trainSize]
    y_train = y[:trainSize]

    x_test = x[trainSize:]
    y_test = y[trainSize:]

    return x_train, y_train, x_test, y_test

# def multilayer_perceptron(x, weights, biases, keep_prob):
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#     return out_layer


# def setup(train_x, train_y):
#     n_hidden_1 = 38
#     n_input = train_x.shape[1]
#     n_classes = train_y.shape[1]
#
#     weights = {
#         'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1])),
#         'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes]))
#     }
#
#     biases = {
#         'b1': tf.Variable(tf.random.normal([n_hidden_1])),
#         'out': tf.Variable(tf.random.normal([n_classes]))
#     }
#
#     keep_prob = tf.Variable("float")
#
#     x = tf.Variable("float", [None, n_input])
#     y = tf.Variable("float", [None, n_classes])
#
#     predictions = multilayer_perceptron(x, weights, biases, keep_prob)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#     optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
#     return optimizer,cost


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = getData()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(x_train.shape[1] * 2, activation='relu'),
        tf.keras.layers.Dense(x_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(x_train.shape[1] // 2, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    #Configure input path if desired

    inputPath = rootDir + "savedModel2020-09-24 04:16:02.057166.ckpt"

    if inputPath is not None:
        print("Loading Model From A file")
        model.load_weights(inputPath)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#    loss_fn = tf.keras.losses.MeanSquaredError()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.007)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'])

    epochSize = 5000

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=outputPath,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq='epoch',
                                                     period=25)

    model.fit(x_train, y_train,
              epochs=epochSize,
              batch_size=64,
              validation_data=(x_test, y_test),
              callbacks=[cp_callback]
              )

