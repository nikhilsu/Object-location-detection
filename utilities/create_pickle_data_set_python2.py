import gzip

import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
import os

from sklearn.model_selection import train_test_split

path_to_images = '../utilities/images/'
pkl_gz1 = '../data/pumpkin1.pkl.gz'
pkl_gz2 = '../data/pumpkin2.pkl.gz'


def dump_to_pkl_file(filename, data_dict):
    gzip_file = gzip.open('%s' % filename, 'wb')
    pickle.dump(data_dict, gzip_file, protocol=2)
    gzip_file.close()


def extract_data(images):
    x = []
    y = []
    for image_name in images:
        path = path_to_images + image_name
        np_array_of_image = np.asarray(Image.open(path).convert('RGB'), dtype='float64') / 255
        x.append(np_array_of_image)
        label = [float(c) for c in image_name[:image_name.index('.jpg')].split(',')]
        y.append(np.asarray(label))
    return x, y


all_images = os.listdir(path_to_images)
print('Extracting images from ' + path_to_images)

train_data_images, test_data_images = train_test_split(all_images, test_size=0.1, random_state=42)
train_x, train_y = extract_data(train_data_images)
test_x, test_y = extract_data(test_data_images)

half_of_training_data = int(len(train_x) / 2)
half_of_test_data = int(len(test_x) / 2)

data1 = dict()
data1['train'] = {}
data1['train']['data'] = np.array(train_x)[:half_of_training_data]
data1['train']['label'] = np.array(train_y)[:half_of_training_data]
data1['test'] = {}
data1['test']['data'] = np.array(test_x)[:half_of_test_data]
data1['test']['label'] = np.array(test_y)[:half_of_test_data]

data2 = dict()
data2['train'] = {}
data2['train']['data'] = np.array(train_x)[half_of_training_data:]
data2['train']['label'] = np.array(train_y)[half_of_training_data:]
data2['test'] = {}
data2['test']['data'] = np.array(test_x)[half_of_test_data:]
data2['test']['label'] = np.array(test_y)[half_of_test_data:]

print("Dump to pumpkin1.pkl.gz...")
dump_to_pkl_file(pkl_gz1, data1)
print("Dump to pumpkin2.pkl.gz...")
dump_to_pkl_file(pkl_gz2, data2)

print('done')
