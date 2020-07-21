import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers

def train_test_split_tdnn(x_seq, y_seq, test_size):    
    data_length = x_seq.shape[0]
    data_id     = np.arange(0, data_length)
    #np.random.shuffle(data_id)

    train_id     = data_id[0:int((1-test_size)*data_length)]
    test_id    = data_id[int((1-test_size)*data_length)::]
    
    x_train     = x_seq[train_id]
    y_train     = y_seq[train_id]
    x_test      = x_seq[test_id]
    y_test      = y_seq[test_id]
    
    return x_train, y_train, x_test, y_test