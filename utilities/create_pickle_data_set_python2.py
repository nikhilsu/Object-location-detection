import gzip

import numpy as np
from PIL import Image
import six.moves.cPickle as pickle
import os

from sklearn.model_selection import train_test_split

path_to_images = '../utilities/images/'
pkl_template = '../data/pumpkin{}.pkl.gz'


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

quarter_of_training_data = int(len(train_x) / 4)
quarter_of_test_data = int(len(test_x) / 4)

start = 0
file_index = 1
for index in range(quarter_of_training_data, len(train_x), quarter_of_training_data):
    end = index + 1
    data = dict()
    data['train'] = {}
    data['train']['data'] = np.array(train_x)[start:end]
    data['train']['label'] = np.array(train_y)[start:end]
    data['test'] = {}
    data['test']['data'] = np.array(test_x)[start:end]
    data['test']['label'] = np.array(test_y)[start:end]

    filename = pkl_template.format(file_index)
    print("Dump to {}...".format(filename))
    dump_to_pkl_file(filename, data)
    start = end
    file_index += 1

print('done')
