import gzip
import pickle


class DataSet:
    def __init__(self, location='../data/pumpkin.pkl.gz'):
        with gzip.open(location, 'rb') as f:
            data_set_dict = pickle.load(f)
        self.train_x, self.train_y = data_set_dict['train']['data'], data_set_dict['train']['label']
        self.test_x, self.test_y = data_set_dict['test']['data'], data_set_dict['test']['label']
