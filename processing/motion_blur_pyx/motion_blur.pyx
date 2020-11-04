import os
import cv2
import time

import numpy as np

from skimage.util import random_noise
from skimage.transform import rescale, resize

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) 
def add_motion_blur(img, flow):
    '''
    Input: 
        img: RGB image with shape (H, W, 3) at time t
        flow: a numpy array with shape (H, W, 2) at time t-> t+1
    Output:
        img_blur: the blurred image with shape (H, W, 3)
    '''
    

    img_h, img_w = img.shape[:2]
    
    img_blur = np.copy(img).astype(np.float32)
    img_counter = np.ones((img_h, img_w))
    
    img = np.expand_dims(img.astype(np.float32), axis=2)

    for y0 in range(img_h):
        for x0 in range(img_w):
            
            dx, dy = flow[y0][x0]
            
            # compute the source of this pixel
            xn = np.round(x0 + dx) # x_{t+1} = x_{t} + dx
            yn = np.round(y0 + dy) # y_{t+1} = y_{t} + dy
            
            # make sure the original pixl is in the bound
            xn = max(0, min(xn, img_w - 1))
            yn = max(0, min(yn, img_h - 1))
            
            # the points on the contour from (x_{t}, y_{t}) -> (x_{t+1}, y_{t+1})
            N = int(max(abs(xn - x0), abs(yn - y0))) + 1
            
            x_path = np.linspace(x0, xn, num=N)[0:].astype(np.int32)
            y_path = np.linspace(y0, yn, num=N)[0:].astype(np.int32)
            
            img_blur[y_path, x_path] += img[y0][x0]# .reshape((1, -1))
            img_counter[y_path, x_path] += 1

    cnt = np.expand_dims(img_counter, axis=-1)
    img_blur /= cnt


    return img_blur

