# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:24:51 2020

@author: isfan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
from solve_discontinuity import solve_discontinuity
from train_test_split import train_test_split_tdnn
from sklearn import preprocessing
import os
import time

def create_directory(directory):
    for i in range(len(directory.split('/'))):
        if directory.split('/')[i] != '':
            sub_dic ='/'.join(directory.split('/')[:(i+1)])
            if not os.path.exists(sub_dic):
                os.makedirs(sub_dic)

log_dir = os.path.join("logs_SNPOrientation", "test_data")
create_directory(log_dir)

'''
#####   Getting the data    #####
'''
stored_df_quat = pd.read_csv('20200420_scene(3)_user(1).csv')
train_gyro_data = np.array(stored_df_quat[['angular_vec_x', 'angular_vec_y', 'angular_vec_z']], dtype=np.float32)
train_acce_data = np.array(stored_df_quat[['acceleration_x', 'acceleration_y', 'acceleration_z']], dtype=np.float32)
train_magn_data = np.array(stored_df_quat[['magnetic_x', 'magnetic_y', 'magnetic_z']], dtype=np.float32)
projection_data = np.array(stored_df_quat[['input_projection_left', 'input_projection_top', 'input_projection_right', 'input_projection_bottom']], dtype=np.int64)
train_euler_data = np.array(stored_df_quat[['input_orientation_pitch', 'input_orientation_roll', 'input_orientation_yaw']], dtype=np.float32)
train_time_data = np.array(stored_df_quat['timestamp'], dtype=np.int64)
train_time_data = train_time_data/705600000
train_data_id = stored_df_quat.shape[0]
print('\nSaved data loaded...\n')

'''
#####   TRAINING DATA PREPROCESSING    #####
'''
print('Training data preprocessing is started...')
#convertion

# Remove zero data from collected training data
system_rate = round((train_data_id+1)/float(np.max(train_time_data) - train_time_data[0]))
idle_period = int(2 * system_rate)
train_gyro_data = train_gyro_data[idle_period:train_data_id, :]*180/np.pi
train_acce_data = train_acce_data[idle_period:train_data_id, :]
train_magn_data = train_magn_data[idle_period:train_data_id, :]
train_euler_data = train_euler_data[idle_period:train_data_id, :]
projection_data = projection_data[idle_period:train_data_id]
train_time_data = train_time_data[idle_period:train_data_id]

train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float32), train_alfa_data])

# Calculate the head orientation
train_acce_data = train_acce_data / 9.8
train_magn_data = train_magn_data
train_euler_data = train_euler_data*180/np.pi

train_euler_data = solve_discontinuity(train_euler_data)

# Create data frame of all features and smoothing
sliding_window_time = 100
sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))

# anticipation time
anticipation_time = 300
anticipation_size = int(np.round(anticipation_time * system_rate/1000))
print ('Anticipation Size: ',anticipation_size)


ann_feature = np.column_stack([train_euler_data, 
                               train_gyro_data, 
                               train_acce_data, 
#                               train_magn_data,
                               ])
    
feature_name = ["pitch", "roll,","yaw",  
                "gX", "gY", "gZ", 
                "aX", "aY", "aZ", 
#                'mX', 'mY', 'mZ',
                ]

ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
#ann_feature_df = ann_feature_df.rolling(sliding_window_size, center=True, min_periods=1).mean()


# Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
spv_name = ["pitch", "roll,","yaw"]

input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)
target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)

input_nm = len(input_series_df.columns)
target_nm = len(target_series_df.columns)

train_length = int(len(ann_feature_df)*0.3)

DELAY_SIZE = int(25 * (system_rate / 250))
TEST_SIZE = 0.5 
TRAIN_SIZE = 1 - TEST_SIZE
print ('Delay Size: ',DELAY_SIZE)

# Variables
TRAINED_MODEL_NAME = './best_net'

# Import datasets
input_series = np.array(input_series_df)
target_series = np.array(target_series_df)

normalizer = preprocessing.StandardScaler()
normalizer.fit(input_series)
input_norm = normalizer.transform(input_series)

# Split training and testing data
x_train, t_train, x_test, t_test = train_test_split_tdnn(input_series, target_series, TEST_SIZE)


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "plot_x"))


class GPCurvesReader(object):
  """Preparing dataset to SNP.

  Supports vector inputs (x) and vector outputs (y). Input are some parameters that lag anticipation time to output.
  """

  def __init__(self,
               batch_size,
               max_num_context,
               orientation,
               len_gen,
               seq_delay,
               data_length,
               testing=False):

    self._batch_size = batch_size
    self._max_num_context = max_num_context
    self._testing = testing
    self._orientation = orientation
    self._data_length = data_length
    self._seq_delay = seq_delay
    self._len_gen  = len_gen

  def NP_dataset(self, x, y, seq_len, con_size):
    if self._orientation == "pitch" or self._orientation == "all":
        ori = 0
    elif self._orientation == "roll":
        ori = 1
    elif self._orientation == "yaw":
        ori = 2
    

    if self._testing and self._orientation != "all":
        x_values = tf.expand_dims(x[:,ori], axis=0)
        for m in range(3):
            if (m!=ori):
                x_values = tf.concat([x_values,tf.expand_dims(x[:,m], axis=0)], axis=0)
    else:
        x_values = tf.expand_dims(x[:,ori], axis=0)
        for m in range(self._batch_size):
            if (m!=ori):
                x_values = tf.concat([x_values,tf.expand_dims(x[:,m], axis=0)], axis=0)
    
    if (self._orientation == "all"):
        y_values_base = tf.expand_dims(y[:,ori], axis=0)
        for m in range(3):
            if (m!=ori):
                y_values_base = tf.concat([y_values_base,tf.expand_dims(y[:,m], axis=0)], axis=0)
        #Repeat the value multiple times
        m = 1
        y_values = y_values_base
        while m<(self._batch_size//3):
            y_values = tf.concat([y_values,y_values_base], axis=0)
            m+=1
    else:
        y_values = tf.tile( tf.expand_dims(y[:,ori], axis=0), [self._batch_size, 1])
    
    x_values = tf.cast(tf.expand_dims(x_values, axis=-1), dtype =tf.float32)
    y_values = tf.cast(tf.expand_dims(y_values, axis=-1), dtype =tf.float32)
    
#    x_axis = tf.tile( tf.expand_dims(np.array(range(len(x))).reshape(-1,1)[:,0], axis=0), [self._batch_size, 1])
    x_axis = tf.tile( tf.expand_dims(tf.range(tf.shape(x)[0]), axis=0), [self._batch_size, 1])
    x_axis = tf.cast(tf.expand_dims(x_axis, axis=-1), dtype =tf.float32)
    
    num_context = tf.random_uniform(
        shape=[], minval=1, maxval= self._max_num_context, dtype=tf.int32)
    
    if self._testing:
      num_target = tf.shape(x_values)[1]
      num_total_points = num_target
      # Select the targets
      target_x = x_values
      target_y = y_values

      idx = tf.random_shuffle(tf.range(num_target))
      if seq_len>int(self._len_gen*con_size):
          num_context = tf.constant(0)
      context_x = tf.gather(x_values, idx[:num_context], axis=1)
      context_y = tf.gather(y_values, idx[:num_context], axis=1)
      plot_x = tf.gather(x_axis, idx[:num_context], axis=1)
    else:
      num_target = tf.random_uniform(shape=(), minval=0, 
               maxval=self._data_length - num_context, dtype=tf.int32)
      num_total_points = num_context + num_target
      # Select the targets which will consist of the context points as well as
      # some new target points
      target_x = x_values[:, :num_total_points, :]
      target_y = y_values[:, :num_total_points, :]
      
      # Select the observations
      context_x = x_values[:, :num_context, :]
      context_y = y_values[:, :num_context, :]
      plot_x = x_axis[:, :num_context, :]
      
    query = ((context_x, context_y), target_x)
    
    return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=tf.shape(target_x)[1],
        num_context_points=num_context,
        plot_x = plot_x)
    
  def SNP_dataset(self, x, y, con_size=1, seed=None):
    context_x_list, context_y_list = [], []
    target_x_list, target_y_list = [], []
    num_total_points_list = []
    num_context_points_list = []
    plot_x = []
    curve_list = []
    
#    Sequential data
    DELAY = tf.constant(self._seq_delay)
#    tf.set_random_seed(self._seed)
    if seed != None:
        i = tf.constant(seed)
    else:
        i = tf.random_uniform(shape=[], minval=0, maxval= len(x)-(DELAY*10)+self._data_length, dtype=tf.int32)
        
    m = 0
    while m <= self._len_gen:
        idx = tf.range(i, tf.math.add(i,tf.constant(self._data_length)))
        curve_list.append(self.NP_dataset(tf.gather(tf.constant(x),idx, axis=0),tf.gather(tf.constant(y),idx, axis=0), m, con_size))
        i = tf.math.add(i,DELAY)
        m+=1
    
    for t in range(len(curve_list)):
        (context_x, context_y), target_x = curve_list[t].query
        target_y = curve_list[t].target_y
        num_total_points_list.append(curve_list[t].num_total_points)
        num_context_points_list.append(curve_list[t].num_context_points)
        plot_x.append(curve_list[t].plot_x)
        context_x_list.append(context_x)
        context_y_list.append(context_y)
        target_x_list.append(target_x)
        target_y_list.append(target_y)

    query = ((context_x_list, context_y_list), target_x_list)

    return NPRegressionDescription(
            query=query,
            target_y=target_y_list,
            num_total_points=num_total_points_list,
            num_context_points=num_context_points_list,
            plot_x = plot_x)


# utility methods
def batch_mlp(input, output_sizes, variable_scope):

    # Get the shapes of the input and reshape to parallelise across
    # observations
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(
                tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(
            output, output_sizes[-1], name="layer_{}".format(i + 1))

    # Bring back into original shape
    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))

    return output

class DeterministicEncoder(object):
  """The Deterministic Encoder."""

  def __init__(self, output_sizes, attention):
    self._output_sizes = output_sizes
    self._attention = attention

  def __call__(self, context_x, context_y, target_x=None, drnn_h=None,
               num_con=None, num_tar=None,
               get_hidden=False, given_hidden=None):

    if given_hidden is None:
        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes,
                        "deterministic_encoder")

        # get hidden
        if get_hidden:
            return hidden

    else:
        hidden = given_hidden

    # Apply attention
    with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
        hidden = self._attention(context_x, target_x, hidden)

    if drnn_h is None:
        hidden = tf.cond(tf.equal(num_con, 0),
                lambda: tf.zeros([target_x.shape[0],
                                num_tar,
                                hidden.shape[-1]]),
                lambda: hidden)
    else:
        drnn_h = tf.tile(tf.expand_dims(drnn_h,axis=1),
                         [1, num_tar, 1])
        hidden = tf.cond(tf.equal(num_con, 0),
                lambda: drnn_h,
                lambda: hidden+drnn_h)

    return hidden

class LatentEncoder(object):
  """The Latent Encoder."""

  def __init__(self, output_sizes, num_latents):

    self._output_sizes = output_sizes
    self._num_latents = num_latents

  def __call__(self, x, y, vrnn_h=None, num=None,
               get_hidden=False, given_hidden=None, irep=None):

    if given_hidden is None:
        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes,
                        "latent_encoder")

        # only get hidden
        if get_hidden:
            return hidden

    else:
        hidden = given_hidden

    # Aggregator: take the mean over all points
    hidden = tf.reduce_mean(hidden, axis=1)

    # when no data, hidden is equal to 0
    hidden = tf.cond(tf.equal(num, 0),
            lambda: tf.zeros([x.shape[0], self._num_latents]),
            lambda: hidden)

    # temporal or not
    if vrnn_h is not None:
        hidden += vrnn_h

    # Have further MLP layers that map to the parameters
    # of the Gaussian latent
    with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
      # First apply intermediate relu layer
      hidden = tf.nn.relu(
          tf.layers.dense(hidden,
                          (self._output_sizes[-1] +
                           self._num_latents)/2,
                          name="penultimate_layer"))
      # Then apply further linear layers to output latent mu
      # and log sigma
      mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
      log_sigma = tf.layers.dense(hidden, self._num_latents,
                                  name="std_layer")

    return mu, log_sigma

class Decoder(object):
  """The Decoder."""

  def __init__(self, output_sizes):

    self._output_sizes = output_sizes

  def __call__(self, representation, target_x):

    # concatenate target_x and representation
    hidden = tf.concat([representation, target_x], axis=-1)

    # Pass final axis through MLP
    hidden = batch_mlp(hidden, self._output_sizes, "decoder")

    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden, 2, axis=-1)

    # Bound the variance
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

    # Get the distribution
    dist = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return dist, mu, sigma

# Performance optimized version
class LatentModel(object):
    """The (A)NP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, deterministic_encoder_output_sizes=None,
                attention=None, beta=1.0, dataset='gp'):

        # encoders
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                            num_latents)
        self._deterministic_encoder = DeterministicEncoder(
            deterministic_encoder_output_sizes, attention)

        # decoder
        self._decoder = Decoder(decoder_output_sizes)

        self._beta = beta

        # to make seen / unseen plots
        self._index = tf.constant(np.arange(0,2000))

    def __call__(self, query, num_targets, num_contexts, target_y,
                inference=True):

        (context_x, context_y), target_x = query

        # get hidden first
        cont_x = tf.concat(context_x,axis=1)
        cont_y = tf.concat(context_y,axis=1)
        prior_h = self._latent_encoder(cont_x, cont_y, get_hidden=True)
        det_h = self._deterministic_encoder(cont_x, cont_y, get_hidden=True)
        tar_x = tf.concat(target_x,axis=1)
        tar_y = tf.concat(target_y,axis=1)
        post_h = self._latent_encoder(tar_x, tar_y, get_hidden=True)

        mu_list, sigma_list = [], []
        log_p_list, kl_list = [], []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_con, log_p_w_con = 0, 0
        mse_list, mse_wo_con, mse_w_con = [], 0, 0
        cnt_wo, cnt_w = tf.constant(0.0), tf.constant(0.0)
        for t in range(len(context_x)):
            cont_x = tf.concat(context_x[:(t+1)],axis=1)
            cont_y = tf.concat(context_y[:(t+1)],axis=1)
            n_con = np.sum(num_contexts[:(t+1)])
            tar_x = tf.concat(target_x[:(t+1)],axis=1)
            tar_y = tf.concat(target_y[:(t+1)],axis=1)
            n_tar = np.sum(num_targets[:(t+1)])

            #########################################
            # latent encoding
            #########################################
            prior_mu, prior_log_sigma = self._latent_encoder(cont_x, cont_y, num=n_con,
                                            given_hidden=prior_h[:,:n_con])
            prior_sigma = tf.exp(0.5*prior_log_sigma)
            prior_latent_rep = prior_mu + prior_sigma*tf.random_normal(tf.shape(prior_mu),0,1,dtype=tf.float32)

            post_mu, post_log_sigma = self._latent_encoder(tar_x, tar_y, num=n_tar,
                                            given_hidden=post_h[:,:n_tar])
            post_sigma = tf.exp(0.5*post_log_sigma)
            post_latent_rep = post_mu + post_sigma*tf.random_normal(tf.shape(post_mu),0,1,dtype=tf.float32)

            if not inference:
                latent_rep = prior_latent_rep
            else:
                latent_rep = post_latent_rep

            latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                                [1, num_targets[t], 1])

            #########################################
            # det encoding
            #########################################
            deterministic_rep = self._deterministic_encoder(cont_x, cont_y,
                                target_x[t], num_con=n_con,
                                num_tar=num_targets[t],
                                given_hidden=det_h[:,:n_con,:])

            #########################################
            # representation making
            #########################################
            representation = tf.concat([deterministic_rep, latent_rep],
                                        axis=-1)

            #########################################
            # decoding
            #########################################
            dist, mu, sigma = self._decoder(representation, target_x[t])
            mu_list.append(mu)
            sigma_list.append(sigma)

            #########################################
            # calculating loss
            #########################################
            log_p = dist.log_prob(target_y[t])

            kl = -0.5*(-prior_log_sigma+post_log_sigma-(tf.exp(post_log_sigma)+(post_mu-prior_mu)**2) / tf.exp(prior_log_sigma) + 1.0)
            kl = tf.reduce_sum(kl, axis=-1, keepdims=True)

            log_p_seen.append(-1*tf.gather(log_p,
                                        self._index[:num_contexts[t]], axis=1))
            log_p_unseen.append(-1*tf.gather(log_p,
                                    self._index[num_contexts[t]:log_p.shape[1]],
                                    axis=1))
            log_p = -tf.reduce_mean(log_p)
            log_p_list.append(log_p)
            kl_list.append(tf.reduce_mean(kl))
            log_p_wo_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:log_p,
                                    lambda:tf.constant(0.0))
            cnt_wo += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(1.0),
                                    lambda:tf.constant(0.0))
            log_p_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:log_p)
            cnt_w += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:tf.constant(1.0))
            mse = tf.losses.mean_squared_error(target_y[t], mu,
                                        reduction=tf.losses.Reduction.NONE)
            mse = tf.reduce_mean(mse)
            mse_list.append(mse)
            mse_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:mse,
                                    lambda:tf.constant(0.0))
            mse_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:mse)

        #########################################
        # results merging
        #########################################
        mu = mu_list
        sigma = sigma_list

        log_p = np.sum(log_p_list) / len(log_p_list)
        log_p_seen = tf.concat(log_p_seen,axis=-1)
        log_p_unseen = tf.concat(log_p_unseen,axis=-1)
        log_p_seen = tf.reduce_mean(log_p_seen)
        log_p_unseen = tf.reduce_mean(log_p_unseen)
        log_p_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: log_p_w_con / cnt_w)
        log_p_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: log_p_wo_con / cnt_wo)
        kl = np.sum(kl_list) / len(kl_list)
        loss = log_p + self._beta * kl
        mse = np.sum(mse_list) / len(mse_list)
        mse_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: mse_w_con / cnt_w)
        mse_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: mse_wo_con / cnt_wo)

        debug_metrics = (log_p, log_p_list, log_p_w_con, log_p_wo_con,
                        mse, mse_list, mse_w_con, mse_wo_con, log_p_seen,
                        log_p_unseen)

        return mu, sigma, log_p, kl, loss, debug_metrics

class TemporalLatentModel(object):
    """The SNP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, deterministic_encoder_output_sizes=None,
                attention=None, beta=1.0, alpha=0.0, dataset='gp'):

        # encoders
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes,
                                            num_latents)
        self._deterministic_encoder = DeterministicEncoder(
            deterministic_encoder_output_sizes, attention)
        # decoder
        self._decoder = Decoder(decoder_output_sizes)

        # rnn modules
        self._drnn = tf.contrib.rnn.BasicLSTMCell(num_latents)
        self._vrnn = tf.contrib.rnn.BasicLSTMCell(num_latents)

        self._beta = beta

        # to make unseen/seen plots
        self._index = tf.constant(np.arange(0,2000))

    def __call__(self, query, num_targets, num_contexts, target_y,
                inference=True):

        (context_x, context_y), target_x = query

        len_seq = len(context_x)

        batch_size = context_x[0].shape[0]

        # latent rnn state initialization
        init_vrnn_state = self._vrnn.zero_state(batch_size, dtype=tf.float32)
        init_vrnn_hidden = init_vrnn_state[1]
        vrnn_state = init_vrnn_state
        vrnn_hidden = init_vrnn_hidden
        latent_rep = init_vrnn_hidden

        # det rnn state initialization
        init_drnn_state = self._drnn.zero_state(batch_size,
                                                dtype=tf.float32)
        init_drnn_hidden = init_drnn_state[1]
        drnn_state = init_drnn_state
        drnn_hidden = init_drnn_hidden
        avg_det_rep = init_drnn_hidden

        mu_list, sigma_list = [], []
        log_p_list, kl_list = [], []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_con, log_p_w_con = 0, 0
        mse_list, mse_wo_con, mse_w_con = [], 0, 0
        cnt_wo, cnt_w = tf.constant(0.0), tf.constant(0.0)
        for t in range(len_seq):

            # current observations
            cont_x = context_x[t]
            cont_y = context_y[t]
            tar_x = target_x[t]

            ########################################
            # latent encoding
            ########################################
            lat_en_c = self._latent_encoder(cont_x, cont_y, get_hidden=True)
            lat_en_c_mean = tf.reduce_mean(lat_en_c, axis=1)

            latent_rep = tf.cond(tf.equal(num_contexts[t],0),
                                lambda: latent_rep,
                                lambda: latent_rep + lat_en_c_mean)

            vrnn_hidden, vrnn_state = self._vrnn(latent_rep, vrnn_state)

            prior_mu, prior_log_sigma = self._latent_encoder(cont_x, cont_y, vrnn_hidden,
                                            num_contexts[t], given_hidden=lat_en_c)
            prior_sigma = tf.exp(0.5*prior_log_sigma)
            tar_y = target_y[t]
            post_mu, post_log_sigma = self._latent_encoder(tar_x, tar_y, vrnn_hidden,
                                                num_targets[t])
            post_sigma = tf.exp(0.5*post_log_sigma)
            prior_latent_rep = prior_mu + prior_sigma*tf.random_normal(tf.shape(prior_mu),0,1,dtype=tf.float32)
            post_latent_rep = post_mu + post_sigma*tf.random_normal(tf.shape(post_mu),0,1,dtype=tf.float32)
            if not inference:
                latent_rep = prior_latent_rep
            else:
                latent_rep = post_latent_rep

            latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                                [1, num_targets[t], 1])

            ########################################
            # det encoding
            ########################################
            det_en_c = self._deterministic_encoder(cont_x, cont_y, get_hidden=True)
            det_en_c_mean = tf.reduce_mean(det_en_c, axis=1)

            avg_det_rep = tf.cond(tf.equal(num_contexts[t],0),
                                lambda: avg_det_rep,
                                lambda: avg_det_rep + det_en_c_mean)
            drnn_hidden, drnn_state = self._drnn(avg_det_rep, drnn_state)

            deterministic_rep = self._deterministic_encoder(cont_x,
                                                            cont_y,
                                                            tar_x,
                                                            drnn_hidden,
                                                            num_con=num_contexts[t],
                                                            num_tar=num_targets[t],
                                                            given_hidden=det_en_c)
            avg_det_rep = tf.reduce_mean(deterministic_rep, axis=1)

            ########################################
            # representation merging
            ########################################
            representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
            latent_rep = tf.reduce_mean(latent_rep, axis=1)

            ########################################
            # decoding
            ########################################
            dist, mu, sigma = self._decoder(representation, tar_x)
            mu_list.append(mu)
            sigma_list.append(sigma)

            ########################################
            # calculating loss
            ########################################
            log_p = dist.log_prob(tar_y)
            log_p_seen.append(-1*tf.gather(log_p,
                                        self._index[:num_contexts[t]], axis=1))
            log_p_unseen.append(-1*tf.gather(log_p,
                                        self._index[num_contexts[t]:log_p.shape[1]],
                                        axis=1))
            kl = -0.5*(-prior_log_sigma+post_log_sigma-(tf.exp(post_log_sigma)+(post_mu-prior_mu)**2) / tf.exp(prior_log_sigma) + 1.0)
            kl = tf.reduce_sum(kl, axis=-1, keepdims=True)
            log_p = -tf.reduce_mean(log_p)
            log_p_list.append(log_p)
            kl_list.append(tf.reduce_mean(kl))
            log_p_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:log_p,
                                    lambda:tf.constant(0.0))
            cnt_wo += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(1.0),
                                    lambda:tf.constant(0.0))
            log_p_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:log_p)
            cnt_w += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:tf.constant(1.0))
            mse = tf.losses.mean_squared_error(target_y[t], mu,
                                        reduction=tf.losses.Reduction.NONE)
            mse = tf.reduce_mean(mse)
            mse_list.append(mse)
            mse_wo_con += tf.cond(tf.equal(num_contexts[t],0), lambda:mse,
                                    lambda:tf.constant(0.0))
            mse_w_con += tf.cond(tf.equal(num_contexts[t],0),
                                    lambda:tf.constant(0.0),
                                    lambda:mse)

        ########################################
        # result merging
        ########################################
        mu = mu_list
        sigma = sigma_list

        log_p = np.sum(log_p_list) / len(log_p_list)
        log_p_seen = tf.concat(log_p_seen,axis=-1)
        log_p_unseen = tf.concat(log_p_unseen,axis=-1)
        log_p_seen = tf.reduce_mean(log_p_seen)
        log_p_unseen = tf.reduce_mean(log_p_unseen)
        log_p_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: log_p_w_con / cnt_w)
        log_p_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: log_p_wo_con / cnt_wo)
        kl = np.sum(kl_list) / len(kl_list)
        log_p = log_p
        kl = kl
        loss = log_p + self._beta * kl

        mse = np.sum(mse_list) / len(mse_list)
        mse_w_con = tf.cond(tf.equal(cnt_w,0), lambda:tf.constant(0.0),
                            lambda: mse_w_con / cnt_w)
        mse_wo_con = tf.cond(tf.equal(cnt_wo,0), lambda:tf.constant(0.0),
                            lambda: mse_wo_con / cnt_wo)

        debug_metrics = (log_p, log_p_list, log_p_w_con, log_p_wo_con,
                        mse, mse_list, mse_w_con, mse_wo_con, log_p_seen,
                        log_p_unseen)

        return mu, sigma, log_p, kl, loss, debug_metrics

def uniform_attention(q, v):
  """Uniform attention. Equivalent to np.

  Args:
    q: queries. tensor of shape [B,m,d_k].
    v: values. tensor of shape [B,n,d_v].

  Returns:
    tensor of shape [B,m,d_v].
  """
  total_points = tf.shape(q)[1]
  rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B,1,d_v]
  rep = tf.tile(rep, [1, total_points, 1])
  return rep

def laplace_attention(q, k, v, scale, normalise):
  """Computes laplace exponential attention.

  Args:
    q: queries. tensor of shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    tensor of shape [B,m,d_v].
  """

  k = tf.expand_dims(k, axis=1)  # [B,1,n,d_k]
  q = tf.expand_dims(q, axis=2)  # [B,m,1,d_k]
  unnorm_weights = - tf.abs((k - q) / scale)  # [B,m,n,d_k]
  unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: 1 + tf.tanh(x)
  weights = weight_fn(unnorm_weights)  # [B,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
  return rep


def dot_product_attention(q, k, v, normalise):
  """Computes dot product attention.

  Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    tensor of shape [B,m,d_v].
  """
  d_k = tf.shape(q)[-1]
  scale = tf.sqrt(tf.cast(d_k, tf.float32))
  unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = tf.sigmoid
  weights = weight_fn(unnorm_weights)  # [B,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
  return rep


def multihead_attention(q, k, v, num_heads=8):
  """Computes multi-head attention.

  Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    num_heads: number of heads. Should divide d_v.

  Returns:
    tensor of shape [B,m,d_v].
  """
  d_k = q.get_shape().as_list()[-1]
  d_v = v.get_shape().as_list()[-1]
  head_size = d_v / num_heads
  key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
  value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
  rep = tf.constant(0.0)
  for h in range(num_heads):
    o = dot_product_attention(
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wq%d' % h, use_bias=False, padding='VALID')(q),
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wk%d' % h, use_bias=False, padding='VALID')(k),
        tf.layers.Conv1D(head_size, 1, kernel_initializer=key_initializer,
                   name='wv%d' % h, use_bias=False, padding='VALID')(v),
        normalise=True)
    rep += tf.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                      name='wo%d' % h, use_bias=False, padding='VALID')(o)
  return rep

class Attention(object):
  """The Attention module."""

  def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
               num_heads=8):
    """Create attention module.

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.
    Args:
      rep: transformation to apply to contexts before computing attention.
          One of: ['identity','mlp'].
      output_sizes: list of number of hidden units per layer of mlp.
          Used only if rep == 'mlp'.
      att_type: type of attention. One of the following:
          ['uniform','laplace','dot_product','multihead']
      scale: scale of attention.
      normalise: Boolean determining whether to:
          1. apply softmax to weights so that they sum to 1 across context pts or
          2. apply custom transformation to have weights in [0,1].
      num_heads: number of heads for multihead.
    """
    self._rep = rep
    self._output_sizes = output_sizes
    self._type = att_type
    self._scale = scale
    self._normalise = normalise
    if self._type == 'multihead':
      self._num_heads = num_heads

  def __call__(self, x1, x2, r):
    """Apply attention to create aggregated representation of r.

    Args:
      x1: tensor of shape [B,n1,d_x].
      x2: tensor of shape [B,n2,d_x].
      r: tensor of shape [B,n1,d].

    Returns:
      tensor of shape [B,n2,d]

    Raises:
      NameError: The argument for rep/type was invalid.
    """
    if self._rep == 'identity':
      k, q = (x1, x2)
    elif self._rep == 'mlp':
      # Pass through MLP
      k = batch_mlp(x1, self._output_sizes, "attention")
      q = batch_mlp(x2, self._output_sizes, "attention")
    else:
      raise NameError("'rep' not among ['identity','mlp']")

    if self._type == 'uniform':
      rep = uniform_attention(q, r)
    elif self._type == 'laplace':
      rep = laplace_attention(q, k, r, self._scale, self._normalise)
    elif self._type == 'dot_product':
      rep = dot_product_attention(q, k, r, self._normalise)
    elif self._type == 'multihead':
      rep = multihead_attention(q, k, r, self._num_heads)
    else:
      raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                       ",'multihead']"))

    return rep

# ## Ploting function
# 
# Same plotting function as for the [implementation of CNPs](https://github.com/deepmind/conditional-neural-process/blob/master/conditional_neural_process.ipynb) that plots the intermediate predictions every so often during training.

def reordering(whole_query, target_y, pred_y, std_y, plot_x, temporal=False):

    (context_x, context_y), target_x = whole_query

    if not temporal:
        for i in range(len(context_x)):
            context_x[i] = context_x[i][:,:,:-1]
        target_x = np.array(target_x)[:,:,:,:-1]

    context_x_list = context_x
    context_y_list = context_y
    target_x_list = target_x
    target_y_list = target_y
    pred_y_list = pred_y
    std_y_list = std_y
    plot_x_list = plot_x

    return (target_x_list, target_y_list, context_x_list, context_y_list,
           pred_y_list, plot_x_list, std_y_list)
    
def plot_functions(plot_data, orientation):
  """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
    plot_x: contains the x value for plottinf purposes
  """
  target_x, target_y, context_x, context_y, pred_y, plot_x, std = plot_data
  if orientation == "all":
      err = np.zeros(3)
      err_99 = np.zeros(3)
      err_pitch_99 = []
      err_roll_99 = []
      err_yaw_99 = []
      '''PLOT PITCH'''
      plt.figure(figsize=(6.4, 4.8*len(target_x)))
      for t in range(len(target_x)):
          plt.subplot(len(target_x),1,t+1)
          # Plot everything
          plt.plot(np.array(range(len(target_y[t][0]))).reshape(-1,1), target_y[t][0], 'k:', linewidth=2)
          plt.plot(np.array(range(len(pred_y[t][0]))).reshape(-1,1), pred_y[t][0], 'b', linewidth=2)
          if len(plot_x[t]) != None:
              plt.plot(plot_x[t][0], context_y[t][0], 'ko', markersize=10)
          plt.fill_between(
              np.array(range(len(target_y[t][0]))),
              pred_y[t][0, :, 0] - std[t][0, :, 0],
              pred_y[t][0, :, 0] + std[t][0, :, 0],
              alpha=0.2,
              facecolor='#65c9f7',
              interpolate=True)
          plt.legend(['Actual', 'Predicted', 'context'])
          plt.grid()
          ax = plt.gca()
          
          err[0]+=np.nanmean(np.abs(np.abs(pred_y[t][0,:,0] - target_y[t][0,:,0])), axis=0)
          err_pitch_99.append(np.nanpercentile(np.abs(np.abs(pred_y[t][0,:,0] - target_y[t][0,:,0])), 99))
          
    #  plt.show()
      plt.savefig(os.path.join(log_dir,'[pitch] img for it-'+str(it)+'.png'))
      plt.close()
      '''PLOT ROLL'''
      plt.figure(figsize=(6.4, 4.8*len(target_x)))
      for t in range(len(target_x)):
          plt.subplot(len(target_x),1,t+1)
          # Plot everything
          plt.plot(np.array(range(len(target_y[t][1]))).reshape(-1,1), target_y[t][1], 'k:', linewidth=2)
          plt.plot(np.array(range(len(pred_y[t][1]))).reshape(-1,1), pred_y[t][1], 'b', linewidth=2)
          if len(plot_x[t]) != None:
              plt.plot(plot_x[t][1], context_y[t][1], 'ko', markersize=10)
          plt.fill_between(
              np.array(range(len(target_y[t][1]))),
              pred_y[t][1, :, 0] - std[t][1, :, 0],
              pred_y[t][1, :, 0] + std[t][1, :, 0],
              alpha=0.2,
              facecolor='#65c9f7',
              interpolate=True)
          plt.legend(['Actual', 'Predicted', 'context'])
          plt.grid()
          ax = plt.gca()
          
          err[1]+=np.nanmean(np.abs(np.abs(pred_y[t][1,:,0] - target_y[t][1,:,0])), axis=0)
          err_roll_99.append(np.nanpercentile(np.abs(np.abs(pred_y[t][1,:,0] - target_y[t][1,:,0])), 99))
    #  plt.show()
      plt.savefig(os.path.join(log_dir,'[roll] img for it-'+str(it)+'.png'))
      plt.close()
      '''PLOT YAW'''
      plt.figure(figsize=(6.4, 4.8*len(target_x)))
      for t in range(len(target_x)):
          plt.subplot(len(target_x),1,t+1)
          # Plot everything
          plt.plot(np.array(range(len(target_y[t][2]))).reshape(-1,1), target_y[t][2], 'k:', linewidth=2)
          plt.plot(np.array(range(len(pred_y[t][2]))).reshape(-1,1), pred_y[t][2], 'b', linewidth=2)
          if len(plot_x[t]) != None:
              plt.plot(plot_x[t][2], context_y[t][2], 'ko', markersize=10)
          plt.fill_between(
              np.array(range(len(target_y[t][2]))),
              pred_y[t][2, :, 0] - std[t][2, :, 0],
              pred_y[t][2, :, 0] + std[t][2, :, 0],
              alpha=0.2,
              facecolor='#65c9f7',
              interpolate=True)
          plt.legend(['Actual', 'Predicted', 'context'])
          plt.grid()
          ax = plt.gca()

          err[2]+=np.nanmean(np.abs(np.abs(pred_y[t][2,:,0] - target_y[t][2,:,0])), axis=0)
          err_yaw_99.append(np.nanpercentile(np.abs(np.abs(pred_y[t][2,:,0] - target_y[t][2,:,0])), 99))
    #  plt.show()
      plt.legend(['Actual', 'Predicted', 'context'])
      plt.savefig(os.path.join(log_dir,'[yaw] img for it-'+str(it)+'.png'))
      plt.close()
      
      err_99 = [np.max(err_pitch_99), np.max(err_roll_99), np.max(err_yaw_99)]
      
      print('MAE Test loss[Pitch, Roll, Yaw]: {:.3f}, {:.3f}, {:.3f}'.format(err[0]/(t+1), err[1]/(t+1), err[2]/(t+1)))
      print('99% error Test loss[Pitch, Roll, Yaw]: {:.3f}, {:.3f}, {:.3f}'.format(err_99[0],err_99[1],err_99[2]))
  else:
      err = 0
      err_99 = []
      plt.figure(figsize=(6.4, 4.8*len(target_x)))
      for t in range(len(target_x)):
          plt.subplot(len(target_x),1,t+1)
          # Plot everything
          plt.plot(np.array(range(len(target_y[t][0]))).reshape(-1,1), target_y[t][0], 'k:', linewidth=2)
          plt.plot(np.array(range(len(pred_y[t][0]))).reshape(-1,1), pred_y[t][0], 'b', linewidth=2)
          if len(plot_x[t]) != None:
              plt.plot(plot_x[t][0], context_y[t][0], 'ko', markersize=10)
          plt.fill_between(
              np.array(range(len(target_y[t][0]))),
              pred_y[t][0, :, 0] - std[t][0, :, 0],
              pred_y[t][0, :, 0] + std[t][0, :, 0],
              alpha=0.2,
              facecolor='#65c9f7',
              interpolate=True)
          plt.legend(['Actual', 'Predicted', 'context'])
          plt.grid()
          ax = plt.gca()
          
          err+=np.nanmean(np.abs(np.abs(pred_y[t][0,:,0] - target_y[t][0,:,0])), axis=0)
          err_99.append(np.nanpercentile(np.abs(np.abs(pred_y[t][0,:,0] - target_y[t][0,:,0])), 99))
      plt.savefig(os.path.join(log_dir,'[' +str(orientation)+'] img for it-'+str(it)+'.png'))
      plt.close()
      
      err_99 = np.max(err_99)
      
      print('MAE Test loss[' +str(orientation)+']: {:.3f}'.format(err/(t+1)))
      print('99% error Test loss[' +str(orientation)+']: {:.3f}'.format(err_99))
      
  return err/(t+1), err_99

'''
#####   SETTING SNP PARAMETERS    #####
'''
TRAINING_ITERATIONS = 5000 #@param {type:"number"}
data_length = 500 #@param [type:"number"]
MAX_CONTEXT_POINTS = int(0.4*data_length)  #@param {type:"number"}
PLOT_AFTER = 1000 #@param {type:"number"}
HIDDEN_SIZE = 128 #@param {type:"number"}
len_gen = 15 #@param [type:"number"]
con_size = 0.5 #@param [type:"number"]
seed_test = 400 #@param [type:"number"]
seq_delay = DELAY_SIZE 
MODEL_TYPE = "SNP" #@param ["NP","SNP"]
orientation = "all" #@param ["all", "pitch","roll", "yaw"]

random_kernel_parameters=True #@param {type:"boolean"}

tf.reset_default_graph()
tf.set_random_seed(2)

beta = tf.placeholder(tf.float32, shape=[])  

## Time Series Input
#x_train = np.tile(np.arange(len(x_train)),(x_train.shape[1],1)).T.astype(np.float64)
#normalizer.fit(x_train)
#x_train = normalizer.transform(x_train)
#
#x_test = np.tile(np.arange(len(x_test)),(x_test.shape[1],1)).T.astype(np.float64)
#normalizer.fit(x_test)
#x_test = normalizer.transform(x_test)

# Train dataset
dataset_train = GPCurvesReader(
    batch_size=3, max_num_context=MAX_CONTEXT_POINTS, orientation = orientation, len_gen = len_gen, seq_delay = seq_delay, data_length = data_length)
data_train = dataset_train.SNP_dataset(x_train,t_train)

# Test dataset
dataset_test = GPCurvesReader(
    batch_size=3, max_num_context=MAX_CONTEXT_POINTS, orientation = orientation, len_gen = len_gen, seq_delay = seq_delay, data_length = data_length, testing=True)
data_test = dataset_test.SNP_dataset(x_test,t_test, con_size = con_size, seed = seed_test)

latent_encoder_output_sizes = [HIDDEN_SIZE]*4
num_latents = HIDDEN_SIZE
deterministic_encoder_output_sizes= [HIDDEN_SIZE]*4
decoder_output_sizes = [HIDDEN_SIZE]*2 + [2]

# SNP with multihead attention
if MODEL_TYPE == 'SNP':
#  attention = Attention(rep='mlp', output_sizes=[HIDDEN_SIZE]*2, att_type='multihead')
  attention = Attention(rep='identity', output_sizes=[HIDDEN_SIZE], att_type='uniform')
  # Define the model
  model = TemporalLatentModel(latent_encoder_output_sizes, num_latents,
                              decoder_output_sizes,
                              deterministic_encoder_output_sizes,attention,
                              beta)
# NP - equivalent to uniform attention
elif MODEL_TYPE == 'NP':
  attention = Attention(rep='identity', output_sizes=None, att_type='uniform')
  # Define the model
  model = LatentModel(latent_encoder_output_sizes, num_latents,
                      decoder_output_sizes,
                      deterministic_encoder_output_sizes,attention,
                      beta)
else:
  raise NameError("MODEL_TYPE not among ['SNP,'NP']")


# Reconstruction
_, _, _, _, loss, _ = model(data_train.query,
                                    data_train.num_total_points,
                                    data_train.num_context_points,
                                    data_train.target_y,
                                    inference=True)

# Generation
mu, sigma, _, _, _, _ = model(data_test.query,
                                        data_test.num_total_points,
                                        data_test.num_context_points,
                                        data_test.target_y,
                                        inference=False)

# Set up the optimizer and train step
optimizer = tf.train.AdamOptimizer(1e-3)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()

#saver = tf.train.Saver()
#checkpoint_path = os.path.join(log_dir, "model")
#ckpt = tf.train.get_checkpoint_state(log_dir)

'''
#####   TRAINING THE SNP MODEL    #####
'''
st = time.time()
# Train and plot
with tf.Session() as sess:
# Initiallize the graph
  sess.run(init)
    
## Continue learning
#  saver = tf.train.import_meta_graph('model-'+str(0)+'.meta')
#  saver.restore(sess,tf.train.latest_checkpoint('./'))

  for it in range(TRAINING_ITERATIONS):

    sess.run([train_step], feed_dict={beta:1.0})

    # Plot the predictions in `PLOT_AFTER` intervals
    if it % PLOT_AFTER == 0 or it == (TRAINING_ITERATIONS-1):
      
      [loss_, pred_y, std_y,
       query, target_y, plot_x] = sess.run([loss, mu, sigma,
                                             data_test.query,
                                             data_test.target_y,
                                             data_test.plot_x],
                                             feed_dict={
                                             beta:1.0}
                                             )
      ft = time.time()-st
      print('\nIteration: {}\nDuration: {:.2f}\nELBO Train loss: {:.3f}'.format(it, ft, loss_))
      
      plot_data = reordering(query, target_y, pred_y, std_y, plot_x, temporal=True)
      err, err_99 = plot_functions(plot_data, orientation)
      
#      if (it!=0):
#          saver.save(sess, './model', global_step=it)

'''
#####   TESTING THE PRE-TRAINED MODEL    #####
'''
#mod_it = 4000
#y_plot = np.zeros(3)
#coordinate = 'yaw'
#print('\nRealtime Plot')
#with tf.Session() as new_sess:
#      loader = tf.train.import_meta_graph('model-'+str(mod_it)+'.meta')
#      loader.restore(new_sess,tf.train.latest_checkpoint('./'))
#      
#      len_gen = 0
#      i = 0
#      dataset_test = GPCurvesReader(
#        batch_size=3, max_num_context=2, orientation = orientation, len_gen = len_gen, seq_delay = 1, data_length = data_length, testing=True)
#      
#      import  RealTimePlot_SNP as rtp
#      
#      st = time.time()
#      for it in range(100):
#          data_test = dataset_test.SNP_dataset(x_test,t_test, seed = it*(seq_delay*len_gen+data_length))
#          mu, sigma, _, _, t_loss, _ = model(data_test.query,
#                                            data_test.num_total_points,
#                                            data_test.num_context_points,
#                                            data_test.target_y,
#                                            inference=False)
#          [loss_, pred_y, std_y,
#           query, target_y, plot_x] = new_sess.run([t_loss, mu, sigma,
#                                                 data_test.query,
#                                                 data_test.target_y,
#                                                 data_test.plot_x],
#                                                 feed_dict={
#                                                 beta:1.0}
#                                                 )
#          ft = time.time()-st
#          print('\nTest Iteration: {}\nDuration: {:.2f}\nELBO Test loss: {:.3f}'.format(it, ft, loss_))
#          
#          
#          (context_x, context_y), target_x = query
#          for lenG in range(len_gen+1):
#             if lenG==(len_gen):
#                plot_len = data_length
#             else:
#                plot_len = seq_delay
#             for seqD in range(plot_len):
#                timestamp_plot_onedata = i/60
#                if coordinate == 'pitch':
#                    y_plot[0] = target_y[lenG][0][seqD]
#                    y_plot[1] = pred_y[lenG][0][seqD]
#                    if seqD in plot_x[lenG][0][:]:
#                        idx = np.where(plot_x[lenG][0][:]==seqD)
#                        y_plot[2] = context_y[lenG][0][idx]
#                    else:
#                        y_plot[2] = 0
#                    
#                elif coordinate == 'roll':
#                    y_plot[0] = target_y[lenG][1][seqD]
#                    y_plot[1] = pred_y[lenG][1][seqD]
#                    if seqD in plot_x[lenG][1][:]:
#                        idx = np.where(plot_x[lenG][1][:]==seqD)
#                        y_plot[2] = context_y[lenG][1][idx]
#                    else:
#                        y_plot[2] = 0
#                elif coordinate == 'yaw':
#                    y_plot[0] = target_y[lenG][2][seqD]
#                    y_plot[1] = pred_y[lenG][2][seqD]
#                    if seqD in plot_x[lenG][2][:]:
#                        idx = np.where(plot_x[lenG][2][:]==seqD)
#                        y_plot[2] = context_y[lenG][2][idx]
#                    else:
#                        y_plot[2] = 0
#                i+=1
#                rtp.RealTimePlot(float(timestamp_plot_onedata), y_plot)