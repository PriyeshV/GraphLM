from __future__ import generators
import arff
import collections
import numpy as np
from tensorflow.python.framework import dtypes
from Utils import ptb_iterator
from vocab import Vocab

class DataSet(object):

  def __init__(self,x,vocab,dtype=dtypes.float32):
    """Construct a DataSet.
    """
    #Add vocab to the dataset
    self.vocab = vocab
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._x = x

  def next_batch(self, batch_size, num_steps,shuffle=True):
    """Return the next `batch_size` examples from this data set.
       Takes Y in one-hot encodings only"""
    for x,y in ptb_iterator(self._x, batch_size, num_steps,shuffle):
        yield x,y

def get_ptb_dataset(dataset):
    for line in open(dataset):
      #Starting and ending is the same symbol
      yield '<eos>'
      for word in line.split():
        if word == '0':
          yield '<eos>'
        #[FIXED]: else statement missing?
        else:
          yield word
        #yield '<eos>'

def read_data_sets(data_dir, dtype=dtypes.float32,validation_ratio=0.20):
    vocab = Vocab()
    vocab.construct(get_ptb_dataset(data_dir+'p_BCDW_walks.txt'))
#    train_x = np.array([vocab.encode(word) for word in get_ptb_dataset(data_dir+'train_BCDW_walks.txt')],dtype=np.int32)
    train_x = np.array([vocab.encode(word) for word in get_ptb_dataset(data_dir+'p_BCDW_walks.txt')],dtype=np.int32)
    validation_x = np.array([vocab.encode(word) for word in get_ptb_dataset(data_dir+'val_BCDW_walks.txt')],dtype=np.int32)

    train = DataSet(train_x, vocab, dtype=dtype)
    validation = DataSet(validation_x, vocab, dtype=dtype)

    datasets_template = collections.namedtuple('Datasets_template', ['train','validation'])
    Datasets = datasets_template(train=train,validation=validation)

    return Datasets

def load_ptb_data():
  return read_data_sets(train_dir=None)

#IMPROVEMENTS REQUIRED
# ADD LINK CHECKING
