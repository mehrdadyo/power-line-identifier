import math
import numpy as np
import torch
import torch.nn as nn
import random
import os, sys
from PIL import Image

def partitionDataset(N, ratio=(90,5,5)):
    'Returns a dictionary ''partititon'' which consists of 3 sets of tuples giving the category and index of image in set'
    # Shuffle data:
    shuf_noUtil = random.sample(range(N['noUtil']),N['noUtil'])
    shuf_noVeg = random.sample(range(N['noVeg']),N['noVeg'])
    shuf_Veg = random.sample(range(N['Veg']),N['Veg'])
    
    # Get number of images in each set for each category:
    train_pct, dev_pct, _ = ratio
    
    index_train_noUtil = math.floor(train_pct/100 * N['noUtil'])
    index_dev_noUtil = math.ceil(dev_pct/100 * N['noUtil']) + index_train_noUtil
    
    index_train_noVeg = math.floor(train_pct/100 * N['noVeg'])
    index_dev_noVeg = math.ceil(dev_pct/100 * N['noVeg']) + index_train_noVeg
    
    index_train_Veg = math.floor(train_pct/100 * N['Veg'])
    index_dev_Veg = math.ceil(dev_pct/100 * N['Veg']) + index_train_Veg
    
    partition = {'train':[],'dev':[],'test':[]}
    # No Utility lists (0):
    for i in range(index_train_noUtil):
        partition['train'].append((0,shuf_noUtil[i]))
        
    for i in range(index_train_noUtil, index_dev_noUtil):
        partition['dev'].append((0,shuf_noUtil[i]))
        
    for i in range(index_dev_noUtil, N['noUtil']):
        partition['test'].append((0,shuf_noUtil[i]))
    
    # No Veg lists (1):
    for i in range(index_train_noVeg):
        partition['train'].append((1,shuf_noVeg[i]))
        
    for i in range(index_train_noVeg, index_dev_noVeg):
        partition['dev'].append((1,shuf_noVeg[i]))
        
    for i in range(index_dev_noVeg, N['noVeg']):
        partition['test'].append((1,shuf_noVeg[i]))
        
    # Veg lists (2):
    for i in range(index_train_Veg):
        partition['train'].append((2,shuf_Veg[i]))
        
    for i in range(index_train_Veg, index_dev_Veg):
        partition['dev'].append((2,shuf_Veg[i]))
        
    for i in range(index_dev_Veg, N['Veg']):
        partition['test'].append((2,shuf_Veg[i]))    
        
    return partition

def resizeImages(folder,size=(224,224)):
    for filename in os.listdir(folder):
        img = Image.open(folder+filename)
        w, h = img.size
        if w > h:
            start_x = int((w-h)/2)
            img = img.crop((start_x,0,start_x+h,h))
        elif h > w:
            start_y = int((h-w)/2)
            img = img.crop((0,start_y,w,start_y+w))
        img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized.save(folder+filename, 'JPEG', quality=90)
        

        
def addWeightsToLayer1(model, model_name):
    layer1_name = list(model.state_dict().keys())[0]
    old_layer1_weights = model.state_dict()[layer1_name]
    d, c, w, h = old_layer1_weights.shape
    old_var = old_layer1_weights.var().item()

    added_layer1_weights = np.random.randn(d,1,h,w)
    added_layer1_weights *= np.sqrt(old_var)   # ensures that the new layer weights are the same order of magnitude as previous
    added_layer1_weights = torch.from_numpy(added_layer1_weights).float()

    new_layer1_weights = torch.cat((old_layer1_weights,added_layer1_weights),1)

    
    # get layer 1 parameters
    try:
        kernel_size = list(model.modules())[1].kernel_size
        stride = list(model.modules())[1].stride
        padding = list(model.modules())[1].padding
        bias = (list(model.modules())[1].bias is not None)
    except AttributeError:
        kernel_size = list(model.modules())[1][0].kernel_size
        stride = list(model.modules())[1][0].stride
        padding = list(model.modules())[1][0].padding
        bias = (list(model.modules())[1][0].bias is not None)
    
    if model_name == 'resnet18':
        model.conv1 = nn.Conv2d(c+1, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        model.features[0] = nn.Conv2d(c+1, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
    model.state_dict()[layer1_name] = new_layer1_weights
    return model
        
def removeImageWeights(model, model_name):
    layer1_name = list(model.state_dict().keys())[0]
    old_layer1_weights = model.state_dict()[layer1_name]
    d, c, w, h = old_layer1_weights.shape
    
    new_layer1_weights = old_layer1_weights[:, 2:]
    
    # get layer 1 parameters
    try:
        kernel_size = list(model.modules())[1].kernel_size
        stride = list(model.modules())[1].stride
        padding = list(model.modules())[1].padding
        bias = (list(model.modules())[1].bias is not None)
    except AttributeError:
        kernel_size = list(model.modules())[1][0].kernel_size
        stride = list(model.modules())[1][0].stride
        padding = list(model.modules())[1][0].padding
        bias = (list(model.modules())[1][0].bias is not None)
        
    if model_name == 'resnet18':
        model.conv1 = nn.Conv2d(c-3, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        model.features[0] = nn.Conv2d(c-3, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
    model.state_dict()[layer1_name] = new_layer1_weights
    return model
        