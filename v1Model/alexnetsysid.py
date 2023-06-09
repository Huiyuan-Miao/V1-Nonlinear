'''
adapted from Cadena et al. 2019 codes
the goal is to make the training process as similar as possible to vgg model in Cadena et al. 2019
'''

import numpy as np
import os
from scipy import stats
import tensorflow as tf

import hashlib
import inspect
import random
from tensorflow.keras import layers
from tensorflow import losses
from collections import OrderedDict
from utils import *
from base_alexnet import Model
import h5py


def readout(inputs, num_neurons, smooth_reg_weight, sparse_reg_weight, group_sparsity_weight):
    s = inputs.shape
    reg = lambda w: l1_regularizer_flatten(w, sparse_reg_weight)
    w_readout = tf.get_variable(
        'w_readout',
        shape=[s[1], num_neurons],
        # initializer = tf.contrib.layers.xavier_initializer(),
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
        regularizer = reg)
    # inputs_ = np.tile(inputs,[1,1,num_neurons])
    predictions = tf.tensordot(inputs, w_readout, [[1],[0]])
    s_ = predictions.shape
    biases = tf.get_variable('biases', shape=[num_neurons], initializer = tf.constant_initializer(value=0.0))
    predictions = predictions + biases
    return predictions


class Alexnet(Model):

    def build(self,
              # name_readout_layer,
              smooth_reg_weight,
              sparse_reg_weight,
              group_sparsity_weight,
              output_nonlin_smooth_weight=1.0,
              b_norm=True):
        self.smooth_reg_weight = smooth_reg_weight # not used actually
        self.sparse_reg_weight = sparse_reg_weight
        self.grout_sparsity_weight = group_sparsity_weight # not used actually

        with self.graph.as_default():
            if b_norm:
                self.feature_bn = tf.layers.batch_normalization(self.alexnetOutput, training = self.is_training,\
                                                         momentum = 0.9, epsilon = 1e-4, name='alexnet_bn', fused =True)
            else:
                self.feature_bn = self.alexnetOutput
            predictions = readout(self.feature_bn, self.data.num_neurons, smooth_reg_weight, sparse_reg_weight,
                                  group_sparsity_weight)
            self.projection = predictions
            self.prediction = tf.nn.elu(self.projection - 1.0) + 1.0
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            self.total_loss = self.get_log_likelihood() + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.initialize()

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.mse, self.prediction]










