import os.path
import random

import numpy as np
import scipy
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential


class CNN:
    """
    CNN classifier
    """

    def __init__(self, x, y, te_x, te_y, epochs=5, batch_size=128):
        """
        initialize CNN classifier
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_x = x
        self.test_x = te_x
        self.train_y = y
        self.test_y = te_y

        # build  CNN model

        self.model = Sequential()

        self.model.add(Conv2D(20, (5, 5), input_shape=(150, 150, 3)))
        self.model.add(Conv2D(20, (5, 5)))
        self.model.add(Conv2D(20, (5, 5)))
        self.model.add(Conv2D(20, (5, 5)))

        # self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        # self.model.add(Dense(64))
        self.model.add(Dense(9))
        self.model.add(Dense(2))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self):
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epochs)

    def evaluate(self):
        predicted = self.model.predict(self.test_x)

        for index, prediction in enumerate(predicted):
            print(self.test_y[index])
            print('predicted')
            print(prediction)

        return self.model.evaluate(self.test_x, self.test_y, verbose=10)


if __name__ == '__main__':

    train_x = []
    train_y = []
    path = "/Users/koushikreddy/Downloads/image"

    list_dir = [i[:-4] for i in os.listdir(path)]
    list_dir.sort()

    for f in list_dir:
        val = []
        for j in f.split(","):
            val.append(j)

        train_y.append(val[0])
        train_y.append(val[2])
        f = str(f) + '.jpg'
        image = scipy.misc.imread(os.path.join(path, f))
        train_x.append(image)

    random.shuffle(train_x)

    train_x = np.array(train_x)
    print(train_x.shape)
    print(train_x[0][0][0][0])
    train_x = train_x / 255
    print(train_x[0][0][0][0])
    print(len(train_x))
    train_y = np.array(train_y).reshape(len(train_x), 2)
    print(train_y.shape)

    cnn = CNN(train_x[:5000], train_y[:5000], train_x[:5001], train_y[:5001])
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
