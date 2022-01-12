from __future__ import print_function, division

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def img2np(images, title, labels,size = (96, 96)):
    counter = 0
    actualImages = []
    labelsDict = {1:"airplane",2:"bird",3:"car",4:"cat",5:"deer",6:"dog",7:"horse",8:"monkey",9:"ship",10:"truck"}
    
    while counter < len(images):
      if labelsDict[labels[counter]] == title:
        actualImages.append(images[counter])
      counter += 1
    # iterating through each file
    for singleImage in actualImages:
        # turn that into a vector / 1D array
        img_ts = [singleImage.ravel()]
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
        except UnboundLocalError: 
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat

def find_mean_img(full_mat):
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(96, 96, 3)
    return mean_img