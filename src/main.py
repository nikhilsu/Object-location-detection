import gzip
import pickle
import numpy as np
import tensorflow as tf

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential


class DataSet:
    def __init__(self, location):
        with gzip.open(location, 'rb') as f:
            data_set_dict = pickle.load(f)
        self.train_x, self.train_y = data_set_dict['train']['data'], data_set_dict['train']['label']
        self.test_x, self.test_y = data_set_dict['test']['data'], data_set_dict['test']['label']


class CNN:
    """
    CNN classifier
    """

    def __init__(self, train_x, train_y, epochs=75, batch_size=128):
        """
        initialize CNN classifier
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y

        # build  CNN model

        self.model = Sequential()
        # self.model.add(MaxPool2D(pool_size=(2, 2),input_shape=(150,150,3)))
        self.model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        # self.model.add(Conv2D(16, (3, 3)))
        # self.model.add(Conv2D(16, (3, 3)))

        self.model.add(Conv2D(8, (3, 3)))
        self.model.add(Conv2D(8, (3, 3)))

        # self.model.add(Conv2D(1, (3, 3)))

        self.model.add(Flatten())
        # self.model.add(Dense(25))
        self.model.add(Dense(3))

        self.model.compile(loss=self.__tukey_bi_weight_loss, optimizer='adam')

    @staticmethod
    def __tukey_bi_weight_loss(y_true, y_predicted):
        z = y_true - y_predicted
        z_abs = tf.abs(z)
        c = 4.685
        subset = tf.cast(tf.less_equal(z_abs, c), z_abs.dtype)
        inv_subset = tf.logical_not(subset)
        c_sq_by_six = c ** 2 / 6
        return (1 - ((1 - ((z / c) ** 2)) ** 3) * subset + inv_subset) * c_sq_by_six

    def train(self):
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epochs)

    def evaluate(self, test_x, test_y, slack_delta=0.15):
        predictions = self.model.predict(test_x)
        number_of_labels = predictions.shape[0]

        correct_predictions = np.less_equal(np.fabs(test_y - predictions), slack_delta)
        accuracy_per_axis = np.sum(correct_predictions, axis=0) / number_of_labels
        accuracy = np.count_nonzero((np.all(correct_predictions, axis=1))) / number_of_labels

        return accuracy, accuracy_per_axis


if __name__ == '__main__':
    data_set = DataSet('../data/pumpkin.pkl.gz')
    print(data_set.train_x.shape)
    print(data_set.train_y.shape)
    sess = tf.Session()
    cnn = CNN(data_set.train_x, data_set.train_y)
    cnn.train()
    accuracies = cnn.evaluate(data_set.test_x, data_set.test_y)
    print('Accuracy of the model: ' + str(accuracies[0]))
    print('Accuracy per axis(x, y, z): ' + str(accuracies[1]))
