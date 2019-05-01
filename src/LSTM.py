import tensorflow as tf
import os

import fourth_version
import fourth_version_polyphonic
import second_version
import third_version
from examples import prepare_examples_with_views

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Blocks warning messages

if __name__ == "__main__":
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # print(device_lib.list_local_devices())

    input_size = 256

    X, y, class_weights = prepare_examples_with_views(input_size)

    # y = to_categorical(y, num_classes=88, dtype='int')

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

    model, callbackItems = fourth_version_polyphonic.get_model_to_train(True)


    print('Finished prep, shape ', X.shape)

    print('Number of training itmes: ', len(X))

    # model.load_weights('fourth_single_output.hdf5')

    model.fit(X, y, epochs=1024, batch_size=64, initial_epoch=0, validation_split=.2, callbacks=callbackItems, class_weight='auto')

    session.close()

