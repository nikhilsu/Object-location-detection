import gzip
import pickle
import numpy as np
from sklearn.utils import shuffle


class DataSet:
    def __init__(self, location1='../data/pumpkin1.pkl.gz', location2='../data/pumpkin2.pkl.gz'):
        with gzip.open(location1, 'rb') as f:
            data_set_dict1 = pickle.load(f)
        with gzip.open(location2, 'rb') as f:
            data_set_dict2 = pickle.load(f)
        self.train_x, self.train_y = shuffle(np.append(data_set_dict1['train']['data'],
                                                       data_set_dict2['train']['data'], axis=0),
                                             np.append(data_set_dict1['train']['label'],
                                                       data_set_dict2['train']['label'], axis=0))

        self.test_x, self.test_y = shuffle(np.append(data_set_dict1['test']['data'],
                                                     data_set_dict2['test']['data'], axis=0),
                                           np.append(data_set_dict1['test']['label'],
                                                     data_set_dict2['test']['label'], axis=0))
