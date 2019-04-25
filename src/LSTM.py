import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Bidirectional
import os
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical

from src.examples import prepare_examples

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Blocks warning messages


if __name__ == "__main__":
    session = tf.Session()

    print('Preparing Examples...')
    X, y = prepare_examples()

    print('X_train:', X)
    print('y_train:', y)
    print(y[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
    y_train = to_categorical(y_train, num_classes=92, dtype='int')
    y_test = to_categorical(y_test, num_classes=92, dtype='int')

    print('AAAAAAAAAAAAA', y_train[0])
    

    
    print('Shape:', X_train.shape[1:])
    print('Shape', X_train.shape)

    inputs = Input(shape=X_train.shape[1:])

    lstm = Bidirectional(LSTM(64), merge_mode='concat')(inputs)
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
        y_train, epochs=1, validation_data=
        (X_test, 
        y_test)
        )

    print(model.predict(np.array([[[72, 64, 0, 0, 0, 0, 0, 0], [72, 69, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [63, 0, 0, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [68, 71, 52, 0, 0, 0, 0, 0], [64, 0, 0, 0, 0, 0, 0, 0], [84, 72, 57, 0, 0, 0, 0, 0]]])))

    session.close()

