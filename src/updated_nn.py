from PIL import Image
from scipy import misc
import glob
import os, os.path
import argparse
import pickle
import gzip
import json
from collections import Counter, defaultdict
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import AvgPool2D
from PIL import Image
import random
from sklearn.utils import shuffle


class test:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 75, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        self.train_x = train_x
        self.test_x  = test_x
        self.train_y = train_y
        self.test_y = test_y

        # build  CNN model

        self.model = Sequential()
        #self.model.add(MaxPool2D(pool_size=(2, 2),input_shape=(150,150,3)))
        self.model.add(Conv2D(32, (3, 3),input_shape=(150,150,3)))
        self.model.add(Conv2D(32, (3, 3)))
	self.model.add(MaxPool2D(pool_size=(2, 2)))
	
	self.model.add(Dropout(0.25))

	self.model.add(Conv2D(16, (3, 3)))
        self.model.add(Conv2D(16, (3, 3)))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

	#self.model.add(Conv2D(16, (3, 3)))
        #self.model.add(Conv2D(16, (3, 3)))

 	self.model.add(Conv2D(8, (3, 3)))
        self.model.add(Conv2D(8, (3, 3)))        

	#self.model.add(Conv2D(1, (3, 3)))

        self.model.add(Flatten())
        #self.model.add(Dense(25))
	self.model.add(Dense(3))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self):

        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y, self.batch_size, self.epoches)

        pass

    def evaluate(self):

        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        predicted = self.model.predict(self.test_x)
        j = 0
	count=0
        countx =0
	county=0
	countz=0
        for i in predicted:
            print(self.test_y[j])
            print('predicted')
            print(i)
	    if((abs(self.test_y[j][0]-i[0]) < 15) and  (abs(self.test_y[j][1]-i[1]) < 15) and (abs(self.test_y[j][2]-i[2]) < 15)):
                count = count +1

	    if(abs(self.test_y[j][0]-i[0]) < 15):
		countx = countx +1
	    if(abs(self.test_y[j][1]-i[1]) < 15):
               	county = county +1
	    if(abs(self.test_y[j][2]-i[2]) < 15):
                countz = countz +1	
            j = j+1
	print(count)
	print(countx)
	print(county)
	print(countz)
	print(len(predicted))
        acc = self.model.evaluate(self.test_x, self.test_y,verbose=10)
        return acc

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='CNN classifier options')
    # parser.add_argument('--limit', type=int, default=-1,
    #                     help='Restrict training to this many examples')
    # args = parser.parse_args()

    train_x = []
    train_y = []
    path = "/home/ironlab/Desktop/machine-learning-csci5622-/project/image"

    list = [i[:-4] for i in os.listdir(path)]
    list.sort()
    fg=0
    for f in list:
        val = []
        for j in f.split(","):
            val.append(j)

        train_y.append(float(val[0]))
        train_y.append(float(val[1]))
        train_y.append(float(val[2]))

        f = str(f)+'.jpg'
        image =  misc.imread(os.path.join(path, f))
	print(fg)
        #for i in range(0, 150):
        #    for j in range(0, 150):
         #       if (image[i][j][0] < 240):
          #          image[i][j][0] = 0
           #         image[i][j][1] = 0
            #        image[i][j][2] = 0
             #   else:
            #        image[i][j][0] = 255
            #        image[i][j][1] = 255
             #       image[i][j][2] = 255
        train_x.append(image)
	fg = fg+1
    print('over')
    train_x = np.array(train_x)

    print(train_x.shape)
    #print(train_x[0][0][0][0])
    train_x = train_x/255
   # print(train_x[0][0][0][0])
    print(len(train_x))
    train_y = np.array(train_y).reshape(len(train_x),3)
    train_y=train_y*100 
    print(train_y.shape)
    train_x,train_y=shuffle(train_x,train_y,random_state=42) 
    print(train_x.shape)
    print(train_y.shape)

    cnn = CNN(train_x[:5000], train_y[:5000], train_x[5001:], train_y[5001:])
    cnn.train()
    acc = cnn.evaluate()
    print(acc)
