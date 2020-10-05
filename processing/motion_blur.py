import os
import cv2
import time

import numpy as np

from skimage.util import random_noise

def add_motion_blur(img, flow):
    '''
    Input: 
        img: RGB image with shape (H, W, 3) at time t
        flow: a numpy array with shape (H, W, 2) at time t-> t+1
    Output:
        img_blur: the blurred image with shape (H, W, 3)
    '''
    
    time_start =time.time()

    img_h, img_w = img.shape[:2]
    
    img_blur = np.copy(img).astype(np.float32)
    img_counter = np.ones((img_h, img_w))
    
    # the longest flow
    flow_length = np.round(np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)).astype(np.int32)

    N = np.amax(flow_length)
    # print('The longest flow is {}'.format(N))
    print('(img_h, img_w): ({}, {})'.format(img_h, img_w))

    for y0 in range(img_h):
        for x0 in range(img_w):
            
            dx, dy = flow[y0][x0]
            
            # compute the source of this pixel
            xn = np.round(x0 + dx) # x_{t+1} = x_{t} + dx
            yn = np.round(y0 + dy) # y_{t+1} = y_{t} + dy
            
            # make sure the original pixl is in the bound
            xn = max(0, min(xn, img_w - 1))
            yn = max(0, min(yn, img_h - 1))
            
            # x0, xn = xn, x0
            # y0, yn = yn, y0

            # the points on the contour from (x_{t}, y_{t}) -> (x_{t+1}, y_{t+1})
            x_path = np.linspace(x0, xn, num = N - 1, endpoint=False)
            y_path = np.linspace(y0, yn, num = N - 1, endpoint=False)

            x_path = np.round(x_path)
            y_path = np.round(y_path)
            
            path = np.array(list(zip(y_path, x_path))).astype(np.int32)
            
            path_new = []
            path_new.append(path[0])
            
            for p in path[1:]:
                if not np.all(p == path_new[-1]): path_new.append(p)
            
            path = np.array(path_new).astype(np.int32)
            
            cnt = np.expand_dims(img_counter[path[:, 0], path[:, 1]], axis=-1)
            
            img_blur[path[:, 0], path[:, 1]] = img_blur[path[:, 0], path[:, 1]] * cnt
            img_blur[path[:, 0], path[:, 1]] += img[y0][x0].reshape((1, -1))
            img_counter[path[:, 0], path[:, 1]] += 1
            img_blur[path[:, 0], path[:, 1]] /= np.expand_dims(img_counter[path[:, 0], path[:, 1]], axis=-1)
            
            '''        
            for idx in range(path.shape[0]):
                
                # if consecutive points on the path is the same
                if (not idx==0) and (np.all(path[idx]==path[idx-1])): continue
                
                y = path[idx][0]
                x = path[idx][1]
                
                # add pixels on (yn, xn) to pixels on the contour
                img_blur[y][x] = img_blur[y][x] * img_counter[y][x] + img[y0][x0]
                img_counter[y][x] += 1
                img_blur[y][x] /= img_counter[y][x]
            '''
    time_end = time.time()

    print('total time: {}'.format(time_end - time_start))
    print('per pixel time: {}'.format((time_end - time_start) / (img_h * img_w)))

    return img_blur

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): # outDir=None, outName="bgr", waitTime=None, angShift=0.0, flagShowFigure=True, 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr

def load_flow(filename):
    return np.load(filename)

def load_rgb(filename):
    return cv2.imread(filename)

def add_simple_motion_blur(img, k_size, horizontal=True):

    kernel = np.zeros((k_size, k_size))
    if horizontal:
        kernel[k_size // 2, :k_size // 2] = 1
    else:
        kernel[:, k_size // 2] = 1
    
    kernel /= np.sum(kernel)

    img_blur = cv2.filter2D(img, -1, kernel)
    
    return img_blur


def customized_salt_and_pepper(img, mode='s&p', ratio=0.03):
    '''
    input: 
        img: shape (H, W, C), with value between [0, 255]
        mode: a string s (salt), p (pepper), s&p (salt and pepper)
        amount: the amount of pixels to add 
    output:
        img_noise: shape (H, W, C)
    '''

    img_h, img_w = img.shape[:2]
    
    num_pixel = img_h * img_w
    num_sample = int(num_pixel * ratio)
    
    img_noise = img

    if 's&p' or 's':
        salt_idx = np.random.randint(0, num_pixel, size=num_sample)
        row = salt_idx // img_w
        col = salt_idx % img_w
        img_noise[row, col, :] = np.array([255, 255, 255])
    
    if 's&p' or 'p':
        pepper_idx = np.random.randint(0, num_pixel, size=num_sample)
        row = pepper_idx // img_w
        col = pepper_idx % img_w
        img_noise[row, col, :] = np.array([0, 0, 0])

    return img_noise
        
def salt_and_pepper_colorful(img, mode, amount):
    
    img_noise = random_noise(img, mode=mode, amount=amount)

    return (img_noise * 255).astype(np.uint8)

def gaussian_noise(img, sigma):

    img = random_noise(img, var=sigma**2)
    
    return (img * 255).astype(np.uint8)

if __name__ == '__main__':
    
    
    # setup directory
    DATA_ROOT = '/data/datasets/wenshanw/tartanair/'
    ENV_NAME = 'abandonedfactory_night'
    LEVEL = 'Easy'
    PATH = 'P000'
    data_dir = os.path.join(DATA_ROOT, ENV_NAME, LEVEL, PATH)
    save_img_path = 'output.png'

    # left image
    left_img1_path = os.path.join(data_dir, 'image_left', '000000_left.png')
    left_img2_path = os.path.join(data_dir, 'image_left', '000001_left.png')
    flow_img_path = os.path.join(data_dir, 'flow', '000000_000001_flow.npy') 
    
    # sample code 
    # imgfile1 = 'data/000380_left.png'
    # imgfile2 = 'data/000381_left.png'
    # flowfile = 'data/000380_000381_flow.npy'

    # img1 = load_rgb(imgfile1)
    # img2 = load_rgb(imgfile2)
    # flow = load_flow(flowfile)
    
    img1 = load_rgb(left_img1_path)
    img2 = load_rgb(left_img2_path)
    flow = load_flow(flow_img_path)
    flowvis = visflow(flow)
    
    cv2.imwrite('img1.png', img1)
    ''' blur image with optical flow''' 
    # img_blur = add_motion_blur(img1.copy(), flow)


    # visimg = np.concatenate((img1,img_blur, img2, flowvis), axis=0)
    # visimg = cv2.resize(visimg, (0, 0), fx=0.5, fy=0.5)
    # cv2.imwrite(save_img_path, visimg)

    ''' add noise to pixels '''
    print(img1.shape)
    # process image with salt and pepper
    img_sp_color = salt_and_pepper_colorful(img1.copy(), mode='s&p', amount=0.03)
    img_sp_bw    = customized_salt_and_pepper(img1.copy(), mode='s&p', ratio=0.015)
    start_time = time.time()
    img_gaussian = gaussian_noise(img1.copy(), 0.1)
    end_time   = time.time()

    print('Gaussian Noise time: {}'.format(end_time - start_time))
    # save image
    cv2.imwrite('salt_pepper_color.png', img_sp_color)
    cv2.imwrite('salt_pepper_bw.png', img_sp_bw)
    cv2.imwrite('gaussian_noise.png', img_gaussian)





