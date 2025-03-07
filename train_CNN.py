''' Summary
This script can be generalized to do any CNN training for object recognition tasks. 
Huiyuan Miao @ 2024
'''

import os
import sys
import random
import time
import numpy
import scipy.io
# import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os, argparse, glob, pickle, subprocess, shlex, io, pprint
from cadena_utils import *
from alexnet import *

parser = argparse.ArgumentParser(description='ImageNet Training')
parser.add_argument('--data_path', default = '/home/tonglab/Datasets/imagenet1000',
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('-o', '--output_path', default='/home/tonglab/Miao/pycharm/',
                    help='path for storing ')
parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate')
parser.add_argument('--step_size', default=60, type=int,
                    help='after how many epochs learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')
FLAGS, FIRE_FLAGS = parser.parse_known_args()

def main(model):

    #### Parameters ####################################################################################################
    model_path = FLAGS.output_path
    train_batch_size = FLAGS.batch_size
    val_batch_size = 32
    start_epoch = 0
    num_epochs = FLAGS.epochs
    save_every_epoch = 60
    initial_learning_rate = FLAGS.lr
    gpu_ids = [1]

    #### Create/Load model #############################################################################################
    # 1. If pre-trained models used without pre-trained weights. e.g., model = models.vgg19()
    # 2. If pre-trained models used with pre-trained weights. e.g., model = models.vgg19(pretrained=True)
    # 3. If our models used.
    ####################################################################################################################

    if len(gpu_ids) > 1:
        # model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0]).cuda()
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, weight_decay=FLAGS.weight_decay)

    #### Resume from checkpoint
    try:
        os.mkdir(model_path)
    except:
        pass

    load_path = os.path.join(model_path, 'checkpoint.pth.tar')
    if os.path.isfile(load_path):
        print("... Loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch'] + 1

        if len(gpu_ids) <= 1: # 1. Multi-GPU to single-GPU or -CPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace('module.', '')  # remove `module.`
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict
        else: # 2. single-GPU or -CPU to Multi-GPU
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'module.' in k:
                    name = k
                else:
                    name = 'module.' + k
                new_state_dict[name] = v
            checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param in optimizer.param_groups:
            param['initial_lr'] = initial_learning_rate
    else:
        print("... No checkpoint found at '{}'".format(load_path))
        print("train from start")


    #### Learning rate scheduler
    lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=FLAGS.step_size,last_epoch=start_epoch-1)
  
    ##### train val dataset - usually only need to change the augmentation method
    train_dataset = torchvision.datasets.ImageFolder(
        # "/home/tonglab/Documents/Data/ILSVRC2012/images/train_16",
        FLAGS.data_path+"/train",
        transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # val_dataset = torchvision.datasets.ImageFolder(
    val_dataset = ImageFolderWithPaths(
        # "/home/tonglab/Documents/Data/ILSVRC2012/images/val_16",
         FLAGS.data_path+"/val",
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.449], std=[0.226])
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    #### Train/Val #####################################################################################################
    for epoch in range(start_epoch,num_epochs):
        if epoch < start_epoch+num_epochs-1:
            # lr_scheduler.step()
            lr.step(epoch=epoch)
        print("... Start epoch at '{}'".format(epoch))
        stat_file = open(os.path.join(model_path, 'training_stats.txt'), 'a+')
        train(train_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids)
        val(val_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids)

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint.pth.tar'))
        if numpy.mod(epoch, save_every_epoch) == save_every_epoch-1:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(model_path, 'checkpoint_epoch_%d.pth.tar'%(epoch)))

def train(train_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids):

    if len(gpu_ids) > 1:
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__

    model.train()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0.

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # print(model.features[0].weight.requires_grad)
        # print(model.features[0].weight[0,0,0,0])
        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        _, num_correct1_batch = is_correct(outputs, targets, topk=1)
        _, num_correct5_batch = is_correct(outputs, targets, topk=5)

        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stat_str = '[Train] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(train_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def val(val_loader, model, loss_function, optimizer, epoch, stat_file, gpu_ids):

    model.eval()
    loss_sum, num_correct1, num_correct5, batch_size_sum = 0., 0., 0., 0.
    correct1, correct5 = [0] * len(val_loader.dataset.samples), [0] * len(val_loader.dataset.samples) # Correct

    for batch_index, (inputs, targets, paths, indices) in enumerate(val_loader):

        if len(gpu_ids) >= 1:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        correct1_batch, num_correct1_batch = is_correct(outputs, targets, topk=1)
        correct5_batch, num_correct5_batch = is_correct(outputs, targets, topk=5)

        loss_sum += loss.item()
        num_correct1 += num_correct1_batch.item()
        num_correct5 += num_correct5_batch.item()
        batch_size_sum += targets.size(0)

        #### Correct
        for i, index in enumerate(indices):
            correct1[index] = correct1_batch.view(-1)[i].item()
            correct5[index] = torch.any(correct5_batch, dim=0).view(-1)[i].item() # top5 glitch

        stat_str = '[Validation] Epoch ({}) LR ({:.4f}) Loss ({:.4f}) Accuracy1 ({:.4f}) Accuracy5 ({:.4f})'.\
            format(epoch, optimizer.param_groups[0]['lr'], loss_sum / (batch_index + 1), num_correct1 / batch_size_sum, num_correct5 / batch_size_sum)
        progress_bar(batch_index, len(val_loader.dataset) / inputs.size(0), stat_str)

    stat_file.write(stat_str + '\n')

def is_correct(output, target, topk=1):
    with torch.no_grad():
        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        num_correct = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True) # top5 glitch

        return correct, num_correct

if __name__ == '__main__':
    for whichModel in [0]:
        if whichModel == 0:
            FLAGS.num_classes = 1000
            FLAGS.batch_size = 256
            FLAGS.weight_decay = 1e-4
            FLAGS.step_size = 60
            FLAGS.data_path = '/home/tonglab/Datasets/imagenet' + str(FLAGS.num_classes)
            FLAGS.output_path = '/home/tonglab/Miao/pycharm/CadenaData_V1/AlexNet_v1_color_' + str(
                FLAGS.num_classes) + 'cate_batchSize' + str(FLAGS.batch_size) + '_wtDecay' + str(FLAGS.weight_decay) +'_stepSize' + str(FLAGS.step_size)
            if os.path.isfile(os.path.join(FLAGS.output_path, 'checkpoint.pth.tar')):
                init_weights = False
            else:
                init_weights = True
                os.makedirs(FLAGS.output_path, exist_ok=True)
            FLAGS.epochs = 180
            model = AlexNet_modified_color(num_classes=FLAGS.num_classes, init_weights=init_weights)
            main(model)
