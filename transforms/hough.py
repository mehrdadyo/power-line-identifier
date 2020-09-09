import numpy as np
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
from skimage.feature import canny

import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
import torch


def hough_transform(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = canny(image)#, 3, 3, 10)
    lines = probabilistic_hough_line(edges, threshold=50, line_length=15,
                                     line_gap=2)

    plt.style.use('dark_background')
    plt.axes().set_aspect('equal')
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))

    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]),'w')
    plt.axis('off')
    plt.close()

    plt.savefig('/home/project/transforms/current.jpg', bbox_inches='tight')

    image = np.max(io.imread('/home/project/transforms/current.jpg').astype(np.uint8), axis=2)[8:232,32:256]
    image = torch.from_numpy(255*(image > 255/2))
    os.remove('/home/project/transforms/current.jpg')
    return image