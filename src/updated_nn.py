import gzip
import pickle

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPool2D
from keras.models import Sequential


class DataSet:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        with gzip.open(location, 'rb') as f:
            data_set_dict = pickle.load(f)
        self.train_x, self.train_y = data_set_dict['train']['data'], data_set_dict['train']['label']
        self.test_x, self.test_y = data_set_dict['test']['data'], data_set_dict['test']['label']


class CNN:
    """
    CNN classifier
    """

    def __init__(self, train_x, train_y, test_x, test_y, epochs=75, batch_size=128):
        """
        initialize CNN classifier
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

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

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epochs)

    def evaluate(self):

        """
        test CNN classifier and get accuracy
        :return: accuracy
        """
        predicted = self.model.predict(self.test_x)
        j = 0
        count = 0
        count_x = 0
        count_y = 0
        count_z = 0
        for i in predicted:
            print(self.test_y[j])
            print('predicted')
            print(i)
            if ((abs(self.test_y[j][0] - i[0]) < 15) and (abs(self.test_y[j][1] - i[1]) < 15) and (
                    abs(self.test_y[j][2] - i[2]) < 15)):
                count = count + 1

            if abs(self.test_y[j][0] - i[0]) < 15:
                count_x = count_x + 1
            if abs(self.test_y[j][1] - i[1]) < 15:
                count_y = count_y + 1
            if abs(self.test_y[j][2] - i[2]) < 15:
                count_z = count_z + 1
            j = j + 1
        print(count)
        print(count_x)
        print(count_y)
        print(count_z)
        print(len(predicted))
        return self.model.evaluate(self.test_x, self.test_y, verbose=10)


if __name__ == '__main__':
    data_set = DataSet('../data/pumpkin.pkl.gz')
    print(data_set.train_x.shape)
    print(data_set.train_y.shape)

    cnn = CNN(data_set.train_x, data_set.train_y, data_set.test_x, data_set.test_y)
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
