import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os
from examples import prepare_examples
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Blocks warning messages


if __name__ == "__main__":
    session = tf.Session()

    print('Preparing Examples...')
    X, y = prepare_examples()

    print('X_train:', X)
    print('y_train:', y)
    print(y[0].shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

    model = Sequential()

    model.add(LSTM(128, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

