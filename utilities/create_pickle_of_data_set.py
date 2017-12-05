import gzip

import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
import os

from sklearn.model_selection import train_test_split

path_to_images = '../utilities/images/'


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

data = dict()
data['train'] = {}
data['train']['data'] = np.array(train_x)
data['train']['label'] = np.array(train_y)
data['test'] = {}
data['test']['data'] = np.array(test_x)
data['test']['label'] = np.array(test_y)

print("Dump to pumpkin.pkl.gz...")
gzip_file = gzip.open('../data/pumpkin.pkl.gz', 'wb')
pickle.dump(data, gzip_file)
gzip_file.close()

print('done')
