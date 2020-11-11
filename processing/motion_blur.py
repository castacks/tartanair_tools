import os
import cv2
import time

import numpy as np

from skimage.util import random_noise
from skimage.transform import rescale, resize

def coordinate_maps(img_h, img_w):
    '''
    Input:
        img_h: the height of the image
        img_w: the width of the image
    Output:
        x_coord_map: x coordinate map with shape (img_h, img_w)
        y_coord_map: y coordinate map with shape (img_h, img_w)s
    '''

    x_coord_map = np.arange(img_w).reshape((1, -1))
    x_coord_map = np.repeat(x_coord_map, img_h, axis=0)

    y_coord_map = np.arange(img_h).reshape((1, -1))
    y_coord_map = np.repeat(y_coord_map, img_w, axis=0).T

    return x_coord_map, y_coord_map

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

    # figure the length of optical flow
    x0_coord_map, y0_coord_map = coordinate_maps(img_h, img_w)
    xn_coord_map = np.clip(np.round(x0_coord_map + flow[:, :, 0]), a_min=0, a_max=img_w-1)
    yn_coord_map = np.clip(np.round(y0_coord_map + flow[:, :, 1]), a_min=0, a_max=img_h-1)
    delta_x = np.abs(xn_coord_map - x0_coord_map)[..., np.newaxis]
    delta_y = np.abs(yn_coord_map - y0_coord_map)[..., np.newaxis]
    coord_map = np.concatenate([delta_x, delta_y], axis=-1).astype(np.int32)
    Ns = np.amax(coord_map, axis=-1) + 1

    max_N = np.amax(Ns)
    min_N = np.amin(Ns)
    dic = {idx: np.arange(idx) for idx in range(min_N, max_N + 1)}

    for y0 in range(img_h):
        for x0 in range(img_w):
            
            xn = xn_coord_map[y0][x0] # x_{t+1} = x_{t} + dx
            yn = yn_coord_map[y0][x0] # y_{t+1} = y_{t} + dy
            N = Ns[y0][x0]

            delta_x = (xn - x0) / (N - 1) if not N == 1 else 0
            delta_y = (yn - y0) / (N - 1) if not N == 1 else 0
            x_path = (dic[N] * delta_x + x0)[0:].astype(np.int32)
            y_path = (dic[N] * delta_y + y0)[0:].astype(np.int32)

            img_blur[y_path, x_path] += img[y0][x0]
            img_counter[y_path, x_path] += 1


    cnt = np.expand_dims(img_counter, axis=-1)
    img_blur /= cnt

    return img_blur

def vector_to_degree(dx, dy):
    # print(dy / (dx + 1e-15))
    angle = np.arctan(dy / (dx + 1e-15)) 
    
    if dx < 0 and dy < 0 and angle >= 0: # the third quadrant
        angle += np.pi
    elif dx < 0 and dy > 0 and angle <= 0:
        angle += np.pi
    elif angle <= 0:
        angle += 2 * np.pi

    return angle

def add_motion_approx(img, flow, num_step=4):
    '''
    Input: 
        img: RGB image with shape (H, W, 3) at time t
        flow: a numpy array with shape (H, W, 2) at time t-> t+1
    Output:
        img_blur: the blurred image with shape (H, W, 3)
    '''
    # compute maximum flow length
    flow_length = np.sum(flow * flow, axis=-1)
    flow_length = np.sqrt(flow_length)
    max_length  = np.amax(flow_length)
    num_step = int(np.ceil(max_length))
    

    img_h, img_w = img.shape[:2]
    img = img.astype(np.float32)
    
    img_blur = np.zeros_like(img)
    blur_maps = {}

    # image move upper left
    steps = np.ones((img_h, img_w, 1))
    upper_left = img.copy()
    blur_maps['upper_left'] = []
    blur_maps['upper_left'].append(img.copy())

    for idx in range(1, num_step+1):

        upper_left[:-idx, :-idx, :] += img[idx:, idx:, :]
        steps[:-idx, :-idx, :] += 1
        
        blur_maps['upper_left'].append(upper_left / steps)

    # image move upper
    steps[:,:, :] = 1
    upper = img.copy()
    blur_maps['upper'] = []
    blur_maps['upper'].append(img.copy())

    for idx in range(1, num_step+1):

        upper[:-idx, :, :] += img[idx:, :, :]
        steps[:-idx, :, :] += 1
        
        blur_maps['upper'].append(upper / steps)


    # image moves upper_right
    steps[:,:,:] = 1
    upper_right = img.copy()
    blur_maps['upper_right'] = []
    blur_maps['upper_right'].append(img.copy())

    for idx in range(1, num_step+1):

        upper_right[:-idx, idx:, :] += img[idx:, :-idx, :]
        steps[:-idx, idx:, :] += 1
        
        blur_maps['upper_right'].append(upper_right / steps)

    # image moves left
    steps[:,:,:] = 1
    left = img.copy()
    blur_maps['left'] = []
    blur_maps['left'].append(img.copy())

    for idx in range(1, num_step+1):
        
        left[:, :-idx, :] += img[:, idx:, :]
        steps[:, :-idx, :] += 1

        blur_maps['left'].append(left / steps)

    # image moves right
    steps[:,:,:] = 1
    right = img.copy()
    blur_maps['right'] = []
    blur_maps['right'].append(img.copy())

    for idx in range(1, num_step+1):

        right[:, idx:, :] += img[:, :-idx, :]
        steps[:, idx:, :] += 1
        blur_maps['right'].append(right / steps)
   
    # image moves lower_left
    steps[:,:,:] = 1
    lower_left = img.copy()
    blur_maps['lower_left'] = []
    blur_maps['lower_left'].append(img.copy())

    for idx in range(1, num_step+1):

        lower_left[idx:, :-idx, :] += img[:-idx, idx:, :]
        steps[idx:, :-idx, :] += 1
    
        blur_maps['lower_left'].append(lower_left / steps)

    # image moves lower
    steps[:,:,:] = 1
    lower = img.copy()
    blur_maps['lower'] = []
    blur_maps['lower'].append(img.copy())
    
    for idx in range(1, num_step+1):

        lower[idx:, :, :] += img[:-idx, :, :]
        steps[idx:, :, :] += 1
    
        blur_maps['lower'].append(lower / steps)

    # image moves lower_right
    steps[:,:,:] = 1
    lower_right = img.copy()
    blur_maps['lower_right'] = []
    blur_maps['lower_right'].append(img.copy())

    for idx in range(1, num_step+1):

        lower_right[idx:, idx:, :] += img[:-idx, :-idx, :]
        steps[idx:, idx:, :] += 1
    
        blur_maps['lower_right'].append(lower_right / steps)

    
    count_n = [0] * 9
    for y in range(img_h):
        for x in range(img_w):
            dx, dy = flow[y, x]
            
            angle = vector_to_degree(dx, dy)
            length = int(np.ceil(np.sqrt(dx * dx + dy * dy)))

            if angle < np.pi / 8 or angle >=  15 / 8 * np.pi:
                img_blur[y][x] = blur_maps['right'][length][y][x] # right[y][x]
                count_n[0] += 1
            elif  np.pi / 8 <= angle < 3 / 8 * np.pi:
                img_blur[y][x] = blur_maps['upper_right'][length][y][x] # upper_right[y][x]
                count_n[1] += 1
            elif 3 / 8 * np.pi <= angle < 5 / 8 * np.pi:
                img_blur[y][x] = blur_maps['upper'][length][y][x] # upper[y][x]
                count_n[2] += 1
            elif 5 / 8 * np.pi <= angle < 7 / 8 * np.pi:
                img_blur[y][x] = blur_maps['upper_left'][length][y][x] # upper_left[y][x]
                count_n[3] += 1
            elif 7 / 8 * np.pi <= angle < 9 / 8 * np.pi:
                img_blur[y][x] = blur_maps['left'][length][y][x] # left[y][x]
                count_n[4] += 1
            elif 9 / 8 * np.pi <= angle < 11 / 8 * np.pi:
                img_blur[y][x] = blur_maps['lower_left'][length][y][x] # lower_left[y][x]
                count_n[5] += 1
            elif 11 / 8 * np.pi <= angle < 13 / 8 * np.pi:
                img_blur[y][x] = blur_maps['lower'][length][y][x] # lower[y][x]
                count_n[6] += 1
            elif 13 / 8 * np.pi <= angle < 15 / 8 * np.pi:
                img_blur[y][x] = blur_maps['lower_right'][length][y][x] # lower_right[y][x]
                count_n[7] += 1
            else:
                print(angle)
            '''
            if dx == 0 and dy == 0: # do not move
                img_blur[y][x] = img[y][x]
                count_n[0] += 1
            elif dx < 0 and dy < 0: # from upper_left
                img_blur[y][x] = upper_left[y][x]
                count_n[1] += 1
            elif dx == 0 and dy < 0: # from upper
                img_blur[y][x] = upper[y][x]
                count_n[2] += 1
            elif dx > 0 and dy < 0: # from upper right
                img_blur[y][x] = upper_right[y][x]
                count_n[3] += 1
            elif dx < 0 and dy == 0: # from left
                img_blur[y][x] = left[y][x]
                count_n[4] += 1
            elif dx > 0 and dy == 0: # from right
                img_blur[y][x] = right[y][x]
                count_n[5] += 1
            elif dx < 0 and dy > 0: # from lower left
                img_blur[y][x] = lower_left[y][x]
                count_n[6] += 1
            elif dx == 0 and dy > 0: # from lower
                img_blur[y][x] = lower[y][x]
                count_n[7] += 1
            elif dx > 0 and dy > 0: # from lower right
                img_blur[y][x] = lower_right[y][x]
                count_n[8] += 1
            '''

    print('count_n: {}'.format(count_n))
            
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
    # left_img1_path = os.path.join(data_dir, 'image_left', '000000_left.png')
    # left_img2_path = os.path.join(data_dir, 'image_left', '000001_left.png')
    # flow_img_path = os.path.join(data_dir, 'flow', '000000_000001_flow.npy') 
    
    # sample code 
    # imgfile1 = 'data/000380_left.png'
    # imgfile2 = 'data/000381_left.png'
    # flowfile = 'data/000380_000381_flow.npy'
    
    # left_img1_path = '/home/chaotec/tartanair_tools/processing/data/000010_left.png'
    # left_img2_path = '/home/chaotec/tartanair_tools/processing/data/000009_left.png'
    # flow_img_path = '/home/chaotec/tartanair_tools/processing/data/000010_000009_flow.npy'

    left_img1_path = './data/000010_left.png'
    left_img2_path = './data/000009_left.png'
    flow_img_path =  './data/000010_000009_flow.npy'

    # img1 = load_rgb(imgfile1)
    # img2 = load_rgb(imgfile2)
    # flow = load_flow(flowfile)
    
    img1 = load_rgb(left_img1_path)
    img2 = load_rgb(left_img2_path)
    flow = load_flow(flow_img_path)
    flowvis = visflow(flow)
    
    cv2.imwrite('img1.png', img1)
    
    ''' blur image with optical flow'''
    print('Time to add motion_blur')
    time_start =time.time()
    
    img_blur = add_motion_blur(img1.copy(), flow)
    # img_blur = add_motion_approx(img1.copy(), flow, num_step=4)
    
    time_end = time.time()
    print('total time: {}'.format(time_end - time_start))
    
    visimg = np.concatenate((img1, img_blur, img2, flowvis), axis=0)
    visimg = cv2.resize(visimg, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(save_img_path, visimg)
    cv2.imwrite('./image_blur.png', img_blur)
    
    # image mask debugging
    # img_diff = np.sum(np.abs(img_blur.astype(np.float32) - img1.astype(np.float32)), axis=-1)
    # mask = img_diff < 2
    # mask = img_diff.astype(np.float32) * 255
    # cv2.imwrite('./mask.png', mask)

    # motion mask
    # flow_magnitude = np.sqrt(flow[:,:,0] * flow[:,:, 0] + flow[:,:,1] * flow[:,:,1])
    # max_v = np.amax(flow_magnitude)
    # min_v = np.amin(flow_magnitude)
    # flow_mask = (flow_magnitude - min_v) / (max_v - min_v) * 255
    # cv2.imwrite('./flow_mask.png', flow_mask)
    
    """
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
    cv2.imwrite('/home/chaotec/tartanair_tools/processing/salt_pepper_color.png', img_sp_color)
    cv2.imwrite('/home/chaotec/tartanair_tools/processing/salt_pepper_bw.png', img_sp_bw)
    cv2.imwrite('/home/chaotec/tartanair_tools/processing/gaussian_noise.png', img_gaussian)
    """
    
