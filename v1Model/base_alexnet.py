'''
Defines the base for a Model for neural system identification
with it's trainer and evaluation functions

-210926
This script is modified for alexnet, the alexnet won't be loaded during the modeling fitting.
Instead, the model layerwise activation will be loaded directly
Thus this script is modified from the Cadena et al. 2019 script "base.py" from https://github.com/sacadena/Cadena2019PlosCB.

-230606
This file can also be used for the modified VGG-19...or any model with pre-saved featuremap from the ProcessFeatureMap

Huiyuan Miao
'''

import numpy as np
import os
from scipy import stats
# import tensorflow as tf
import hashlib
import tensorflow.compat.v1 as tf
import inspect
import random
from tensorflow.keras import layers
from tensorflow import losses
# tf.disable_v2_behavior()

import h5py

class Model:

    def __init__(self, data=None, log_dir=None, log_hash=None, global_step=None, obs_noise_model='poisson',name_readout_layer = None,dataDir = None):
        self.data = data
        log_dir_ = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
        log_dir = os.path.join(log_dir_, 'train_logs', 'cnn_tmp' if log_dir is None else log_dir)
        if log_hash == None: log_hash = '%010x' % random.getrandbits(40)
        self.log_dir = os.path.join(log_dir, log_hash)
        self.log_hash = log_hash
        self.seed = int.from_bytes(log_hash[:4].encode('utf8'), 'big')
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.obs_noise_model = obs_noise_model
        self.best_loss = 1e100
        self.val_iter_loss = []
        self.name_readout_layer = name_readout_layer

        filename = "layer" + str(self.name_readout_layer) + "Processed.h5"
        PATH = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
        alexnetOutputPath = os.path.join(PATH, dataDir+filename)
        with h5py.File(alexnetOutputPath, "r") as f:
            # List all groups
            # print("Keys: %s" % f.keys())
            data_key = list(f.keys())[0]
            # Get the data
            alexnetOutput = list(f[data_key])
            alexnetOutput_ = np.array(alexnetOutput)
            self.alexnetfea = alexnetOutput_
        if data is None: return
        with tf.Graph().as_default() as self.graph:
            self.is_training = tf.placeholder(tf.bool, name = 'is_training')
            self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
            self.images = tf.placeholder(tf.float32, shape=[None, data.px_y, data.px_x, 1], name='images')
            self.responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons], name='responses')
            self.realresp  = tf.placeholder(tf.float32, shape=[None, data.num_neurons], name='realresp')
            self.alexnetOutput = tf.placeholder(tf.float32, shape=[None, np.shape(alexnetOutput)[1]], name='alexnetOutput')

    def initialize(self):
        loss_summ = tf.summary.scalar('loss_function', self.total_loss)
        self.summaries = tf.summary.merge_all()
        # config = tf.ConfigProto(
        #     device_count={"CPU": 6},  # limit to num_cpu_core CPU usage
        # )
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_best = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.log_dir, max_queue=0, flush_secs=0.1)

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
                self.writer.close()
        except:
            pass

    def close(self):
        self.session.close()

    def save(self, step=None):
        if step == None:
            step = self.global_step
        chkp_file = os.path.join(self.log_dir, 'model.ckpt')
        self.saver.save(self.session, chkp_file, global_step=step)

    def save_best(self):
        self.saver_best.save(self.session, os.path.join(self.log_dir, 'best.ckpt'))

    def load(self, step=None):
        if step == None:
            step = self.global_step
        else:
            self.global_step = step
        chkp_file = os.path.join(self.log_dir, 'model.ckpt-%d' % step)
        self.saver.restore(self.session, chkp_file)

    def load_best(self):
        self.saver_best.restore(self.session, os.path.join(self.log_dir, 'best.ckpt'))

    def train(self,
              max_iter=10000,
              learning_rate=0.005,
              batch_size=256,
              val_steps=100,
              save_steps=500,
              early_stopping_steps=5):
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1,:]
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            idx_val, res_val, realresp_val = self.data.val_alexnet()
            input_val = alexnetOutput_[idx_val]
            not_improved = 0
            for i in range(self.global_step + 1, self.global_step + max_iter + 1):
                # print(i)
                # training step
                idx_batch, res_batch, rresp_batch = self.data.minibatch_alexnet(batch_size)
                # print(idx_batch)
                # alexnetOutput_ = alexnetOutput_[self.data.train_idx]
                input_batch = alexnetOutput_[idx_batch]
                self.global_step = i
                # print(input_batch.shape)
                feed_dict = {self.alexnetOutput: input_batch,
                             self.responses: res_batch,
                             self.realresp: rresp_batch,
                             self.is_training: True,
                             self.learning_rate: learning_rate}
                self.session.run([self.train_step, update_ops], feed_dict)

                if not i % save_steps:
                    self.save(i)
                if not i % val_steps:
                    result = self.eval(alexnetoutput=input_val,
                                       responses=res_val,
                                       realresp=realresp_val,
                                       with_summaries=True,
                                       keep_record_loss=True,
                                       global_step=i,
                                       learning_rate=learning_rate)
                    if result[0] < self.best_loss:
                        self.best_loss = result[0]
                        self.save_best()
                        not_improved = 0
                    else:
                        not_improved += 1
                    if not_improved == early_stopping_steps:
                        self.global_step -= early_stopping_steps * val_steps
                        self.load_best()
                        not_improved = 0
                        break
                    yield (i, result[:-1])

    def eval(self, with_summaries=False, keep_record_loss=False, alexnetoutput=None, responses=None, realresp=None,
             global_step=None, learning_rate=None):
        if (alexnetoutput is None) or (responses is None):
            idx, responses, realresp = self.data.test_alexnet()
            nrep, nim, nneu = responses.shape
            idx = np.tile(idx,nrep)
            alexnetoutput = self.alexnetfea[idx-1,:]
            responses = responses.reshape([nim * nrep, nneu])
            realresp = realresp.reshape([nim * nrep, nneu])
        ops = self.get_test_ops()
        feed_dict = {self.alexnetOutput: alexnetoutput,
                     self.responses: responses,
                     self.realresp: realresp,
                     self.is_training: False}
        if with_summaries:
            assert global_step != None, 'global_step must be set for summaries'
            assert learning_rate != None, 'learning_rate must be set for summaries'
            ops += [self.summaries]
            feed_dict[self.learning_rate] = learning_rate
        result = self.session.run(ops, feed_dict)
        if with_summaries: self.writer.add_summary(result[-1], global_step)
        if keep_record_loss: self.val_iter_loss.append(result[0])
        return result

    def compute_log_likelihoods(self, prediction, response, realresp):
        self.poisson = tf.reduce_mean(tf.reduce_sum((prediction - response * tf.log(prediction + 1e-9)) \
                                                    * realresp, axis=0) / tf.reduce_sum(realresp, axis=0))
        self.poisson2 = tf.reduce_mean(tf.reduce_sum((tf.exp(prediction) - response * prediction)\
                                                    * realresp, axis=0) / tf.reduce_sum(realresp,axis=0))
        self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - response) \
                                                * realresp, axis=0) / tf.reduce_sum(realresp, axis=0))

    def get_log_likelihood(self):
        if self.obs_noise_model == 'poisson':
            return self.poisson
        elif self.obs_noise_model == 'gaussian':
            return self.mse
        elif self.obs_noise_model == 'poisson2':
            return self.poisson2

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.prediction]

    def performance_test(self):
        '''
        This function computes the explainable variance explained on the test set
        '''
        idx, responses, real_responses = self.data.test_alexnet()
        nrep, nim, nneu = responses.shape
        alexnetOutput = self.alexnetfea[idx-1, :]

        # get predictions
        predictions_test = self.prediction.eval(session=self.session, \
                                                feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        predictions_test = np.tile(predictions_test.T, nrep).T

        # replace inserted zeros in responses arrays with nans.
        responses_nan = self.data.nanarray(real_responses, responses)

        # mean squared error
        mse = np.nanmean((predictions_test - responses_nan.reshape([nrep * nim, nneu])) ** 2, axis=0)

        total_variance, explainable_var = [], []
        for n in range(self.data.num_neurons):
            rep = self.data.repetitions[n]  # use only original number of repetitions
            resp_ = responses_nan[:rep, :, n]
            obs_var = np.nanmean((np.nanvar(resp_, axis=0, ddof=1)), axis=0)  # obs variance
            tot_var = np.nanvar(resp_, axis=(0, 1), ddof=1)  # total variance
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)  # explainable variance

        total_variance = np.array(total_variance)
        # mse[mse > total_variance] = total_variance[mse > total_variance]
        explainable_var = np.array(explainable_var)
        var_explained = total_variance - mse
        eve = var_explained / explainable_var  # explainable variance explained

        self.eve = eve
        self.var_explained = var_explained
        self.explainable_var = explainable_var
        self.MSE = mse
        self.total_variance = total_variance

    def evaluate_avg_corr_test(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.test_alexnet()
        nrep, nim, nneu = res.shape
        alexnetOutput = self.alexnetfea[idx-1, :]

        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        pred = np.tile(pred.T, nrep).T

        temp = res
        temp[real_res.astype(np.bool) == False] = np.nan
        res_mean_ = np.nanmean(temp,axis = 0)
        res_mean = np.tile(res_mean_, [res.shape[0], 1])

        res = res.reshape(nrep*nim,nneu)
        real_res = real_res.reshape(nrep*nim,nneu)

        # iterate over neurons
        corrs = []
        corrs_mean = []
        eve_r2Based = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r_mean = res_mean[:, i]
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i].astype(np.bool)
            r_mean =  np.compress(b, r_mean)
            r = np.compress(b, r)
            p = np.compress(b, p)
            corr_mean = pearsonr(r, r_mean)[0]
            corr = pearsonr(r, p)[0]

            #             if np.isnan(corr):
            #                 print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)
            corrs_mean.append(corr_mean)
            eve_r2Based.append(corr**2/corr_mean**2)
        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        self.eve_r2Based_test = eve_r2Based
        self.maxCorr_test = np.array([v for v in corrs_mean if not np.isnan(v)])
        return clean_corrs, avg_corr


    def evaluate_avg_corr_imglevel_test(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.test_alexnet()
        nrep, nim, nneu = res.shape
        alexnetOutput = self.alexnetfea[idx-1, :]

        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        res = res.reshape(nrep*nim,nneu)
        real_res = real_res.reshape(nrep*nim,nneu)

        # iterate over neurons
        corrs = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i]
            rep = np.sum(b.reshape(-1,len(p)),axis = 0)
            rep_ = rep.astype(np.bool)
            rep[rep == 0] = 1
            r_mean = np.sum((r*b).reshape(-1,len(p)),axis = 0)/rep
            r_mean = np.compress(rep_, r_mean)
            p = np.compress(rep_,p)
            corr = pearsonr(r_mean[~np.isnan(r_mean)], p[~np.isnan(r_mean)])[0]

            # if np.isnan(corr):
            #      print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)

        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        return clean_corrs, avg_corr

    def performance_val(self):
        '''
        This function computes the explainable variance explained on the validation set
        '''
        idx, responses, real_responses = self.data.val_alexnet()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx,:]
        predictions_val = self.prediction.eval(session=self.session, \
                                               feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        # replace inserted zeros in responses arrays with nans.
        responses_nan = self.data.nanarray(real_responses, responses)

        # mean aquared error
        # mse  = np.nanvar((predictions_val - responses_nan),axis=0)
        mse = np.nanmean((predictions_val - responses_nan) ** 2, axis=0)
        sz = responses_nan.shape[0]
        resps_reshaped = responses_nan.reshape(
            [self.data.num_reps, int(sz / self.data.num_reps), self.data.num_neurons])

        total_variance, explainable_var = [], []
        for n in range(self.data.num_neurons):
            rep = self.data.repetitions[n]  # use only original number of repetitions
            resp_ = resps_reshaped[:rep, :, n]
            obs_var = np.nanmean((np.nanvar(resp_, axis=0, ddof=1)), axis=0)  # obs variance
            tot_var = np.nanvar(resp_, axis=(0, 1), ddof=1)  # total variance
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)  # explainable variance

        total_variance = np.array(total_variance)
        # mse[mse > total_variance] = total_variance[mse > total_variance]
        explainable_var = np.array(explainable_var)
        var_explained = total_variance - mse
        eve = var_explained / explainable_var

        self.eve_val = eve
        self.var_explained_val = var_explained
        self.explainable_var_val = explainable_var
        self.MSE_val = mse
        self.total_variance_val = total_variance

    def evaluate_avg_corr_val(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.val_alexnet()  # e.g. eval_data = data.val()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx, :]

        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        res_ = res.reshape(self.data.num_reps,-1,self.data.num_neurons)
        real_res_ = real_res.reshape(self.data.num_reps,-1,self.data.num_neurons)
        temp = res_
        temp[real_res_.astype(np.bool) == False] = np.nan
        res_mean_ = np.nanmean(temp,axis = 0)
        res_mean = np.tile(res_mean_, [res_.shape[0], 1])
        # iterate over neurons
        corrs = []
        corrs_mean = []
        eve_r2Based = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r_mean = res_mean[:, i]
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i].astype(np.bool)
            r_mean = np.compress(b, r_mean)
            r = np.compress(b, r)
            p = np.compress(b, p)
            corr = pearsonr(r, p)[0]
            corr_mean = pearsonr(r, r_mean)[0]

            #             if np.isnan(corr):
            #                 print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)
            corrs_mean.append(corr_mean)
            eve_r2Based.append(corr ** 2 / corr_mean ** 2)
        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        self.eve_r2Based_val = eve_r2Based
        self.maxCorr_val = np.array([v for v in corrs_mean if not np.isnan(v)])
        return clean_corrs, avg_corr

    def evaluate_avg_corr_imglevel_val(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.val_alexnet()  # e.g. eval_data = data.val()
        idx_val = self.data.image_ids[idx]
        idx_val = np.unique(idx_val)
        alexnetOutput = self.alexnetfea[idx_val-1, :]

        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})

        # iterate over neurons
        corrs = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i]
            rep = np.sum(b.reshape(-1,len(p)),axis = 0)
            rep_ = rep.astype(np.bool)
            rep[rep == 0] = 1
            r_mean = np.sum((r*b).reshape(-1,len(p)),axis = 0)/rep
            r_mean = np.compress(rep_, r_mean)
            p = np.compress(rep_,p)
            corr = pearsonr(r_mean, p)[0]

            # if np.isnan(corr):
            #      print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)

        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        return clean_corrs, avg_corr

    def performance_train(self):
        '''
        This function computes the explainable variance explained on the validation set
        '''
        idx, responses, real_responses = self.data.train_alexnet()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx,:]
        num_idx = len(idx)
        alexnetOutput1 = alexnetOutput[:int(num_idx/4),:]
        alexnetOutput2 = alexnetOutput[int(num_idx/4):int(num_idx/2),:]
        alexnetOutput3 = alexnetOutput[int(num_idx/2):int(num_idx/4*3),:]
        alexnetOutput4 = alexnetOutput[int(num_idx/4*3):,:]
        pred1 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput1, self.is_training: False})
        pred2 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput2, self.is_training: False})
        pred3 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput3, self.is_training: False})
        pred4 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput4, self.is_training: False})
        predictions_val = np.vstack((pred1,pred2,pred3,pred4))
        # replace inserted zeros in responses arrays with nans.
        responses_nan = self.data.nanarray(real_responses, responses)

        # mean aquared error
        # mse  = np.nanvar((predictions_val - responses_nan),axis=0)
        mse = np.nanmean((predictions_val - responses_nan) ** 2, axis=0)
        sz = responses_nan.shape[0]
        resps_reshaped = responses_nan.reshape(
            [self.data.num_reps, int(sz / self.data.num_reps), self.data.num_neurons])

        total_variance, explainable_var = [], []
        for n in range(self.data.num_neurons):
            rep = self.data.repetitions[n]  # use only original number of repetitions
            resp_ = resps_reshaped[:rep, :, n]
            obs_var = np.nanmean((np.nanvar(resp_, axis=0, ddof=1)), axis=0)  # obs variance
            tot_var = np.nanvar(resp_, axis=(0, 1), ddof=1)  # total variance
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)  # explainable variance

        total_variance = np.array(total_variance)
        # mse[mse > total_variance] = total_variance[mse > total_variance]
        explainable_var = np.array(explainable_var)
        var_explained = total_variance - mse
        eve = var_explained / explainable_var

        self.eve_train= eve
        self.var_explained_train = var_explained
        self.explainable_var_train = explainable_var
        self.MSE_train = mse
        self.total_variance_train = total_variance

    def evaluate_avg_corr_train(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.train_alexnet()  # e.g. eval_data = data.val()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx, :]
        num_idx = len(idx)

        alexnetOutput1 = alexnetOutput[:int(num_idx/4),:]
        alexnetOutput2 = alexnetOutput[int(num_idx/4):int(num_idx/2),:]
        alexnetOutput3 = alexnetOutput[int(num_idx/2):int(num_idx/4*3),:]
        alexnetOutput4 = alexnetOutput[int(num_idx/4*3):,:]
        pred1 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput1, self.is_training: False})
        pred2 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput2, self.is_training: False})
        pred3 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput3, self.is_training: False})
        pred4 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput4, self.is_training: False})
        pred = np.vstack((pred1,pred2,pred3,pred4))
        res_ = res.reshape(self.data.num_reps,-1,self.data.num_neurons)
        real_res_ = real_res.reshape(self.data.num_reps,-1,self.data.num_neurons)
        temp = res_
        temp[real_res_.astype(np.bool) == False] = np.nan
        res_mean_ = np.nanmean(temp,axis = 0)
        res_mean = np.tile(res_mean_, [res_.shape[0], 1])
        # iterate over neurons
        corrs = []
        corrs_mean = []
        eve_r2Based = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r_mean = res_mean[:, i]
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i].astype(np.bool)
            r_mean = np.compress(b, r_mean)
            r = np.compress(b, r)
            p = np.compress(b, p)
            corr = pearsonr(r, p)[0]
            corr_mean = pearsonr(r, r_mean)[0]

            #             if np.isnan(corr):
            #                 print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)
            corrs_mean.append(corr_mean)
            eve_r2Based.append(corr ** 2 / corr_mean ** 2)
        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        self.eve_r2Based_train = eve_r2Based
        self.maxCorr_train = np.array([v for v in corrs_mean if not np.isnan(v)])
        return clean_corrs, avg_corr


    def evaluate_avg_corr_imglevel_train(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr

        idx, res, real_res = self.data.train_alexnet()  # e.g. eval_data = data.val()
        idx_train = self.data.image_ids[idx]
        idx_train = np.unique(idx_train)
        alexnetOutput = self.alexnetfea[idx_train-1, :]

        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})

        # iterate over neurons
        corrs = []
        for i in range(self.data.num_neurons):
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i]
            rep = np.sum(b.reshape(-1,len(p)),axis = 0)
            rep_ = rep.astype(np.bool)
            rep[rep == 0] = 1
            r_mean = np.sum((r*b).reshape(-1,len(p)),axis = 0)/rep
            r_mean = np.compress(rep_, r_mean)
            p = np.compress(rep_,p)
            corr = pearsonr(r_mean, p)[0]

            # if np.isnan(corr):
            #      print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)

        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr = np.mean(clean_corrs)
        return clean_corrs, avg_corr

    def evaluationwrapper_test(self):
        self.performance_test()
        eve_test = self.eve
        correlation_testset, avg_correlation_testset = self.evaluate_avg_corr_test()
        correlation_imglevel_testset, avg_correlation_imglevel_testset = self.evaluate_avg_corr_imglevel_test()
        eve_r2Based_test = self.eve_r2Based_test
        maxCorr_test = self.maxCorr_test
        return eve_test, correlation_testset, avg_correlation_testset, correlation_imglevel_testset, avg_correlation_imglevel_testset,eve_r2Based_test,maxCorr_test


    def evaluationwrapper_val(self):
        self.performance_val()
        eve_val = self.eve_val
        correlation_valset, avg_correlation_valset = self.evaluate_avg_corr_val()
        correlation_imglevel_valset, avg_correlation_imglevel_valset = self.evaluate_avg_corr_imglevel_val()
        eve_r2Based_val = self.eve_r2Based_val
        maxCorr_val = self.maxCorr_val
        return eve_val, correlation_valset, avg_correlation_valset, correlation_imglevel_valset, avg_correlation_imglevel_valset,eve_r2Based_val,maxCorr_val

    def evaluationwrapper_train(self):
        self.performance_train()
        eve_train = self.eve_train
        correlation_trainset, avg_correlation_trainset = self.evaluate_avg_corr_train()
        correlation_imglevel_trainset, avg_correlation_imglevel_trainset = self.evaluate_avg_corr_imglevel_train()
        eve_r2Based_train = self.eve_r2Based_train
        maxCorr_train = self.maxCorr_train
        return eve_train, correlation_trainset, avg_correlation_trainset, correlation_imglevel_trainset, avg_correlation_imglevel_trainset,eve_r2Based_train,maxCorr_train

    def saveOutput(self):
        import scipy.io
        # train
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        idx, responses, real_responses = self.data.train_alexnet()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx, :]
        num_idx = len(idx)
        alexnetOutput1 = alexnetOutput[:int(num_idx/4),:]
        alexnetOutput2 = alexnetOutput[int(num_idx/4):int(num_idx/2),:]
        alexnetOutput3 = alexnetOutput[int(num_idx/2):int(num_idx/4*3),:]
        alexnetOutput4 = alexnetOutput[int(num_idx/4*3):,:]
        pred1 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput1, self.is_training: False})
        pred2 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput2, self.is_training: False})
        pred3 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput3, self.is_training: False})
        pred4 = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput4, self.is_training: False})
        pred = np.vstack((pred1,pred2,pred3,pred4))
        # pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        fev, corr_single_trial, avg_corr_single_trial, corr_single_img, avg_corr_single_img,fev_r2Based,maxCorr_single_trial = self.evaluationwrapper_train()
        output_file = os.path.join(self.log_dir, 'TrainPred.mat')
        scipy.io.savemat(output_file, {'Prediction': pred, 'ImageIdx': self.data.train_idx+1, 'FEV': fev,
                                       'corr_single_trial': corr_single_trial,'avg_corr_single_trial':avg_corr_single_trial,
                                       'corr_single_img': corr_single_img, 'avg_corr_single_img': avg_corr_single_img,
                                       'FEV_r2Based':fev_r2Based,'maxCorr_single_trial': maxCorr_single_trial})
        # val
        idx, responses, real_responses = self.data.val_alexnet()
        alexnetOutput_ = self.alexnetfea[self.data.image_ids-1, :]
        alexnetOutput = alexnetOutput_[idx, :]
        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        fev, corr_single_trial, avg_corr_single_trial, corr_single_img, avg_corr_single_img,fev_r2Based,maxCorr_single_trial  = self.evaluationwrapper_val()
        output_file = os.path.join(self.log_dir, 'ValPred.mat')
        scipy.io.savemat(output_file, {'Prediction': pred, 'ImageIdx': self.data.val_idx+1, 'FEV': fev,
                                       'corr_single_trial': corr_single_trial,'avg_corr_single_trial':avg_corr_single_trial,
                                       'corr_single_img': corr_single_img, 'avg_corr_single_img': avg_corr_single_img,
                                       'FEV_r2Based':fev_r2Based,'maxCorr_single_trial': maxCorr_single_trial})
        # test
        idx, responses, real_responses = self.data.test_alexnet()
        alexnetOutput = self.alexnetfea[idx-1, :]
        pred = self.prediction.eval(session=self.session, feed_dict={self.alexnetOutput: alexnetOutput, self.is_training: False})
        fev, corr_single_trial, avg_corr_single_trial, corr_single_img, avg_corr_single_img,fev_r2Based,maxCorr_single_trial  = self.evaluationwrapper_test()
        output_file = os.path.join(self.log_dir, 'TestPred.mat')
        scipy.io.savemat(output_file, {'Prediction': pred, 'ImageIdx': self.data.image_ids_test, 'FEV': fev,
                                       'corr_single_trial': corr_single_trial,'avg_corr_single_trial':avg_corr_single_trial,
                                       'corr_single_img': corr_single_img, 'avg_corr_single_img': avg_corr_single_img,
                                       'FEV_r2Based':fev_r2Based,'maxCorr_single_trial': maxCorr_single_trial})


## ignore the last two parameters...fev_r2Based,maxCorr_single_trial
