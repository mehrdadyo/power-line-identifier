import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

import matplotlib.image as mpimg 
import os, sys
from PIL import Image

import numpy as np
import torch

def hog_transform(img):
    fd, hog_image = hog(img, orientations=16, pixels_per_cell=(8, 8),
                    cells_per_block=(28, 28), visualize=True, multichannel=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 25))
    return torch.from_numpy(hog_image_rescaled)

def hog_transform_batch(img_batch):
    N, C, W, H = img_batch.shape
    hog_img_batch = torch.zeros(N,1,W,H);
    for n in range(N):
        hog_img_batch[n] = hog_transform(img_batch[n].permute(1,2,0))
    new_batch = torch.cat((img_batch,hog_img_batch), dim=1) 
    return new_batch