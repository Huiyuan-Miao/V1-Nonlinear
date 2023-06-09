'''
Those the file title says alexnet, it is actually design to take care of any model with pre-saved feature map from processFeatureMap.py
'''
# % pylab inline
import os
import sys
import numpy as np
import importlib
import inspect

p = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
if p not in sys.path:
    sys.path.append(p)

from data import Dataset, MonkeyDataset
from alexnetsysid import *
import tensorflow.compat.v1 as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus),len(logical_gpus))
    except RuntimeError as e:
        print(e)

def modelFitting(numlayer,rescale,whichModel,whichFeaturemap,sparse_reg_weight):
    log_hash = 'AlexNet/lambda'+str(sparse_reg_weight)+'/rescale'+str(rescale)+'/'+whichModel +'/'+whichFeaturemap
    dataDir = 'AlexNetFeatureMap/rescale'+str(rescale)+'/'+whichModel +'/'+whichFeaturemap+'/'
    for layerNum in (0,numlayer):
        print(layerNum)
        data_dict = Dataset.get_clean_data()
        data = MonkeyDataset(data_dict, seed=1000, train_frac=0.8 ,subsample=2, crop = 30)
        model = Alexnet(data, log_dir='monkey_rescale', log_hash=log_hash+'/layer'+str(layerNum), obs_noise_model='poisson',name_readout_layer = str(layerNum),dataDir = dataDir)

        model.build(
                  smooth_reg_weight=0.1, # this param is not used, has been removed from the training script
                  sparse_reg_weight=sparse_reg_weight,
                  group_sparsity_weight=0.1, # this param is not used, has been removed from the training script
                  output_nonlin_smooth_weight=-1,
                  b_norm=True)
        # Training the network
        learning_rate=1e-4
        for lr_decay in range(3):
            training = model.train(max_iter=10000,
                                 val_steps=100,
                                 save_steps=10000,
                                 early_stopping_steps=10,
                                 batch_size=256,
                                 learning_rate=learning_rate)
            for (i, (logl, total_loss, mse, pred)) in training:
                print('Step %d | Total loss: %s | %s: %s | MSE: %s | Var(y): %s' % (i, total_loss, model.obs_noise_model, logl, mse, np.mean(np.var(pred, axis=0))))
            learning_rate /= 3  # Learning rate decays to one third once it stops improving
            print('Reducing learning rate to %f' % learning_rate)
        model.load_best()
        model.saveOutput()
        print('Done fitting ' + whichModel + ' rescale ' + str(rescale) + ' whichFeaturemap ' + whichFeaturemap + ' layer '+ str(layerNum) )

def runmodelFitting(rescales,whichModels,whichFeaturemaps,numlayers,sparse_reg_weight):
    for rs in range(0,len(rescales)):
        rescale = rescales[rs]
        if os.path.exists('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight)+'/rescale'+str(rescale)) == 0:
            os.mkdir('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight)+'/rescale'+str(rescale))
        for wm in range(0,len(whichModels)):
            whichModel = whichModels[wm]
            numlayer = numlayers[wm]
            if os.path.exists('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight) + '/rescale' + str(rescale)+'/'+whichModel) == 0:
                os.mkdir('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight) + '/rescale' + str(rescale)+'/'+whichModel)
            for wf in range(0,len(whichFeaturemaps)):
                whichFeaturemap = whichFeaturemaps[wf]
                if os.path.exists('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight) + '/rescale' + str(
                        rescale) + '/' + whichModel+'/'+whichFeaturemap) == 0:
                    os.mkdir('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight) + '/rescale' + str(
                        rescale) + '/' + whichModel+'/'+whichFeaturemap)
                modelFitting(numlayer, rescale, whichModel, whichFeaturemap, sparse_reg_weight)
                tf.keras.backend.clear_session()

sparse_reg_weight = 0.03
if os.path.exists('../train_logs/monkey_rescale/AlexNet/lambda'+str(sparse_reg_weight)) == 0:
    os.mkdir('../train_logs/monkey_rescale/AlexNet/lambda' + str(sparse_reg_weight))

rescales = [40];
whichModels = ['AlexNet_modified_color']
whichFeaturemaps = ['featureMap']
numlayers = [17]
runmodelFitting(rescales,whichModels,whichFeaturemaps,numlayers,sparse_reg_weight)
