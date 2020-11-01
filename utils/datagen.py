import os
import tensorflow as tf
import numpy as np

MODEL_INPUT_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'inputfiles'))

if not os.path.exists(MODEL_INPUT_DATAPATH):
    os.makedirs(MODEL_INPUT_DATAPATH)
    
PROCESSED_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'processeddata'))

if not os.path.exists(PROCESSED_DATAPATH):
    os.makedirs(PROCESSED_DATAPATH)
    
RESULTFILE_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'resultfiles'))

if not os.path.exists(RESULTFILE_DATAPATH):
    os.makedirs(RESULTFILE_DATAPATH)
    

strategies = ['original', 'oversample_basic', 'oversample_tte', 'augment_basic', 'augment_tte']
targets = {'hosp_death': 'tt_dth', 'icu_death': 'tt_dth', 'icu_readm': 'tt_readm', 'long_icu': 'duration'}
tables = ['vs_tokens', 'all_tokens']
    
features = {'hosp_death': tf.io.FixedLenFeature([], tf.int64),
            'icu_death': tf.io.FixedLenFeature([], tf.int64),
            'icu_readm': tf.io.FixedLenFeature([], tf.int64),
            'long_icu': tf.io.FixedLenFeature([], tf.int64),
            'tt_dth': tf.io.FixedLenFeature([], tf.int64),
            'tt_readm': tf.io.FixedLenFeature([], tf.int64),
            'duration': tf.io.FixedLenFeature([], tf.int64),
            'hist_icu': tf.io.FixedLenFeature([], tf.int64),
            'ave_dur': tf.io.FixedLenFeature([], tf.int64),
            'time_since': tf.io.FixedLenFeature([], tf.int64),
            'subject': tf.io.FixedLenFeature([], tf.int64),                
            'hosp_adm': tf.io.FixedLenFeature([], tf.int64),
            'data_clin': tf.io.VarLenFeature(dtype=tf.int64),
            'times_clin': tf.io.VarLenFeature(dtype=tf.int64),
            'data_vs': tf.io.VarLenFeature(dtype=tf.int64),
            'times_vs': tf.io.VarLenFeature(dtype=tf.int64)}
    
    
def parse_record(data_record, target_test, model_type):
    
    example = tf.io.parse_single_example(data_record, features=features)
    data_lookup = {'clin': 'data_clin', 'vs': 'data_vs'}[model_type]
    time_to_lookup = targets[target_test]
    X = example[data_lookup]
    Y = example[target_test]#tf.stack([example[target_test], tf.ones_like(example[target_test]) - example[target_test]])
    ID = example['hosp_adm']
    TT = example[time_to_lookup]
    return (tf.sparse.to_dense(X), Y, ID, TT)

def parse_record_both(data_record, target_test):
   
    example = tf.io.parse_single_example(data_record, features=features)
    data_lookup = {'clin': 'data_clin', 'vs': 'data_vs'}
    time_to_lookup = targets[target_test]
    X = [tf.sparse.to_dense(example[d]) for d in data_lookup.values()]
    Y = example[target_test]
    ID = example['hosp_adm']
    TT = example[time_to_lookup]
    return (X[0], X[1], Y, ID, TT)


def get_datafile(target, strategy, phase, fold, batch_size, model_type='clin'):
    files = [os.path.join(MODEL_INPUT_DATAPATH, f) for f in os.listdir(MODEL_INPUT_DATAPATH) if strategy in f and phase in f and str(fold) in f]
    if strategy != 'original':
        files = [f for f in files if target in f]
    dataset = tf.data.TFRecordDataset(files)
    if model_type == 'both':
        dataset = dataset.map(lambda x: parse_record_both(x, target))
    else:
        dataset = dataset.map(lambda x: parse_record(x, target, model_type))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)
    return dataset


def get_embedding_layer(table):
    embedding_weight = np.genfromtxt(os.path.join(RESULTFILE_DATAPATH, f'embedding_vector_{table}'), delimiter='\t')
    max_features = embedding_weight.shape[0]
    output_dim = embedding_weight.shape[1]
    layer = tf.keras.layers.Embedding(max_features, output_dim, trainable=False)
    layer.build((None, max_features))  # replace None with a fixed batch size if appropriate
    layer.set_weights([embedding_weight])
    return layer


class DataGen():
    
    def __init__(self, strategy, target, fold, mode='train', batch_size=256, epochs=10, full_list=True):
        
        self.suffix = targets[target]
        if strategy == 'basic':
            self.datafiles = [os.path.join(MODEL_INPUT_DATAPATH, f'data_{mode}_k_fold_{fold}_{i}_{strategy}') for i in range(10)]
        else:
            self.datafiles = [os.path.join(MODEL_INPUT_DATAPATH, f'data_{mode}_k_fold_{fold}_{i}_{strategy}_{self.suffix}') for i in range(10)]
        if not full_list:
            self.datafiles = self.datafiles[:2]
        self.batch_size = batch_size
        self.epochs = epochs
        self.target = target
        self.mode = mode
        self.fold = fold
        
        for d in self.datafiles:
            if not os.path.exists(d):
                raise(ValueError(f'File {d} not available in target directory'))
            
    def get_iterator(self):
        self.dataset = tf.data.TFRecordDataset(self.datafiles)
        self.dataset = self.dataset.map(lambda x: parse_record(x, self.target))
        self.dataset = self.dataset.repeat(self.epochs)
        self.dataset = self.dataset.shuffle(10000)
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.prefetch(2)
        return iter(self.dataset)
            
        