# an example script for saving features from DNN models in a format that can be passed to the V1 model training script. Can change to any model, but remember to also modify the layer name you would like to save. 
import os
import inspect
import torch
import torch.nn as nn
import collections
import numpy
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
from PIL import Image
from vgg import *
# import tensorflow as tf

DATA_PATH = '../Cadena_VGG-19/Cadena_PlosCB19_data/data_binned_responses/' # downloaded from Cadena et al. (2019) github page https://github.com/sacadena/Cadena2019PlosCB
FILE = 'cadena_ploscb_data.pkl'
file_path = DATA_PATH + FILE
with open(file_path, 'rb') as g:
    data = pickle.load(g)

gpu_ids = [1]
device = torch.device('cuda:%d' % (gpu_ids[0]))
torch.cuda.set_device(device)

def lodeModel(model,load_path,device):
    checkpoint = torch.load(load_path, map_location='cpu')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['state_dict'].items():  # Multi-GPUs to Single-GPU or -CPU
        if 'module.' in k:
            name = k.replace('module.', '')
        else:
            name = k
        new_state_dict[name] = v
    checkpoint['state_dict'] = new_state_dict
    # device = torch.device('cpu')
    model.cuda()
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def lodeModel_untrain(model,device):
    model.cuda()
    model.to(device)
    model.eval()
    return model


rescales = [40] # the input image size to be used
rootDirs = ['vgg19_modified2/'] # which model to be saved
ckptName = 'checkpoint.pth.tar' # the name of the check point


subsample=1
crop=30
for rs in range(0,len(rescales)):
# for rs in range(0,1):
    rescale = rescales[rs]
    saveDir_ = '../Cadena_VGG-19/modifiedVGG19FeatureMap/rescale' + str(rescale) +'/vgg19_modified2'
    if os.path.isdir(saveDir_)==0:
        os.mkdir(saveDir_)
    for rd in [0]:#range(0,len(rootDirs)):
        rootDir = './'+ rootDirs[rd]
        images_ = data['images'][:, crop:-crop:subsample, crop:-crop:subsample, None]
        img = np.zeros((images_.shape[0],rescale,rescale))
        for l in range(0,images_.shape[0]):
            img[l,:,:] = np.array(Image.fromarray(images_[l,:,:,0]).resize((rescale,rescale),Image.BICUBIC))
        imgs_mean = np.mean(img)
        imgs_sd = np.std(img)
        img = (img - imgs_mean) / imgs_sd
        if rd == 0:
            model = vgg19_modified2()
            numLayer =27
            # here are the name of layers that you would like to save
            nm = ['conv1_1','ReLU1_1','conv1_2','ReLU1_2','pool1','conv2_1','ReLU2_1','conv2_2','ReLU2_2','pool2',
    'conv3_1','ReLU3_1','conv3_2','ReLU3_2','conv3_3','ReLU3_3','conv3_4','ReLU3_4','pool3',
    'conv4_1','ReLU4_1','conv4_2','ReLU4_2','conv4_3','ReLU4_3','conv4_4','ReLU4_4','pool4',
    'conv5_1','ReLU5_1','conv5_2','ReLU5_2','conv5_3','ReLU5_3','conv5_4','ReLU5_4','pool5']
            saveDir = saveDir_


        if os.path.isdir(saveDir) == 0:
            os.mkdir(saveDir)

        load_path = rootDir + ckptName
        if rd == 0:
            model = lodeModel(model,load_path,device)
            # model = lodeModel_untrain(model,device)
        # image preprocessing - select based on your need - if your model is trained with RGB colored images
        for i in range(0,29): # process images in batch, reduce memory use 
            print(str(i))
            # for color
            imgInput = torch.zeros([250, 3, rescale, rescale])
            if i == 0:
                img_ = img.reshape(7250, 1, rescale, rescale)
                img = np.tile(img_, (1, 3, 1, 1))
                img = torch.tensor(img, dtype=torch.float32)

            # for gray - select based on your need - if your model is trained with gray scale images with only one channel
            # imgInput = torch.zeros([250, 1, rescale, rescale])
            # if i == 0:
            #     img_ = img.reshape(7250, 1, rescale, rescale)
            #     img = torch.tensor(img_,dtype=torch.float32)

            imgInput = img[i * 250:(i + 1) * 250,:,:,:]

            # imgInput = torch.tensor(imgInput)
            a = model.get_outputs(imgInput.to(device))
            for j in range(0,numLayer):
                featuremap = a[j]
                nm = saveDir + 'layer'+str(j)+'_'+str(i)+'.h5'
                with h5py.File(nm, "w") as data_file:
                    data_file.create_dataset("feature_maps", data=featuremap.astype(np.float16))

        # combine all outputs from the same layer
        for j in range(0,numLayer):
            featuremap = []
            data_processed = list()
            for k in range(0,29):
                nm = saveDir + 'layer'+str(j)+'_'+str(k)+'.h5'
                with h5py.File(nm, "r") as f:
                    a_group_key = list(f.keys())[0]
                    featuremap = np.array(f[a_group_key]).astype(np.float16)
                os.remove(nm)
                d = list(featuremap)
                for i in range(250):
                    temp = d[i]
                    temp2 = temp.reshape(-1)
                    data_processed.append(temp2)
            if os.path.isdir(saveDir+'featureMap') == 0:
                os.mkdir(saveDir+'featureMap')
            savefilename = saveDir + 'featureMap/layer' + str(j) + 'Processed.h5'
            with h5py.File(savefilename, "w") as data_file:
                data_file.create_dataset("feature_maps", data=data_processed)

