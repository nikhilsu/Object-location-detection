import sys

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Activation, AvgPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential

if sys.version_info[0] < 3:
    from data_set_py2 import DataSet
else:
    from data_set_py3 import DataSet


class CNN:
    """
    CNN classifier
    """

    def __init__(self, train_x, train_y, epochs=100, batch_size=128):
        """
        initialize CNN classifier
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y

        # build  CNN model

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        self.model.add(Activation('relu'))
        self.model.add(AvgPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(AvgPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(AvgPool2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(8, (3, 3)))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(3))

        self.model.compile(loss=self.__tukey_bi_weight_loss, optimizer='adam', metrics=['acc'])

    @staticmethod
    def __tukey_bi_weight_loss(y_true, y_predicted):
        z = y_true - y_predicted
        z_abs = tf.abs(z)
        c = 4.685
        subset_bool = tf.less_equal(z_abs, c)
        subset = tf.cast(subset_bool, z_abs.dtype)
        inv_subset = tf.cast(tf.logical_not(subset_bool), z_abs.dtype)
        c_sq_by_six = c ** 2 / 6
        return (1 - ((1 - ((z / c) ** 2)) ** 3) * subset + inv_subset) * c_sq_by_six

    def train(self):
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epochs)

    def evaluate(self, test_x, test_y, slack_delta=0.1):
        predictions = self.model.predict(test_x)
        number_of_labels = float(predictions.shape[0])

        correct_predictions = np.less_equal(np.fabs(test_y - predictions), slack_delta)
        accuracy_per_axis = np.sum(correct_predictions, axis=0) / number_of_labels
        accuracy = np.count_nonzero((np.all(correct_predictions, axis=1))) / number_of_labels

        return accuracy, accuracy_per_axis

    def save_model(self, filename):
        self.model.save(filename)


if __name__ == '__main__':
    model_filename = 'pumpkin_model.h5'
    data_set = DataSet()
    print(data_set.train_x.shape)
    print(data_set.train_y.shape)
    cnn = CNN(np.append(data_set.train_x, data_set.test_x, axis=0),
              np.append(data_set.train_y, data_set.test_y, axis=0))
    cnn.train()
    print("Saving model to " + model_filename)
    cnn.save_model(model_filename)
    accuracies = cnn.evaluate(data_set.test_x, data_set.test_y)
    print('Accuracy of the model: ' + str(accuracies[0]))
    print('Accuracy per axis(x, y, z): ' + str(accuracies[1]))
