import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import skinematics as skin
import matplotlib.pyplot as plt
import math
import collections
#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

def NP_dataset(x, y, batch_size, max_num_context, testing=False):
    
    x_values = tf.tile( tf.expand_dims(x[:,0], axis=0), [batch_size, 1])
    x_values = tf.cast(tf.expand_dims(x_values, axis=-1), dtype =tf.float32)
    
    y_values = tf.tile( tf.expand_dims(y[:,0], axis=0), [batch_size, 1])
    y_values = tf.cast(tf.expand_dims(y_values, axis=-1), dtype =tf.float32)
    
    num_context = tf.random_uniform(
        shape=[], minval=3, maxval= max_num_context, dtype=tf.int32)
    
    if testing:
      num_target = tf.shape(x_values)[1]
      num_total_points = num_target
      # Select the targets
      target_x = x_values
      target_y = y_values

      # Select the observations
      idx = tf.random_shuffle(tf.range(num_target))
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)

    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
               maxval=max_num_context - num_context, dtype=tf.int32)
      num_total_points = num_context + num_target
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_total_points, :]
      target_y = y_values[:, :num_total_points, :]

      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]

    query = ((context_x, context_y), target_x)
    
    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context)