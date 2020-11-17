import os
import cv2
import time
import glob
import argparse
import multiprocessing

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

def test_on_predefined_image():
    
    left_img1_path = './data/000010_left.png'
    left_img2_path = './data/000009_left.png'
    flow_img_path =  './data/000010_000009_flow.npy'

    img1 = load_rgb(left_img1_path)
    img2 = load_rgb(left_img2_path)
    flow = load_flow(flow_img_path)
    flowvis = visflow(flow)
    
    cv2.imwrite('img1.png', img1)
    
    ''' blur image with optical flow'''
    print('Time to add motion_blur')
    time_start =time.time()
    
    img_blur = add_motion_blur(img1.copy(), flow)
    
    time_end = time.time()
    print('total time: {}'.format(time_end - time_start))
    
    visimg = np.concatenate((img1, img_blur, img2, flowvis), axis=0)
    visimg = cv2.resize(visimg, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('./img_collage.png', visimg)
    cv2.imwrite('./image_blur.png', img_blur)
    
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
    
def blur_image_multithread_wrapper(args):
     
    start_time = time.time()
    
    img_path = args[0]
    flow_path = args[1]
    save_dir = args[2]
   
    img_name = img_path.split('/')[-1]
    save_img_path = os.path.join(save_dir, img_name)
    
    if os.path.exists(save_img_path): 
        print('{} exists.'.format(save_img_path))
        return

    img = load_rgb(img_path)
    flow = load_flow(flow_path)
    
    img_blur = add_motion_blur(img, flow)
    cv2.imwrite(save_img_path, img_blur.astype(np.uint8))
    print('Deal with img: {}, take: {} secs and save to {}'.format(img_name, time.time() - start_time, save_img_path))

def arg_parse():
    
    parser = argparse.ArgumentParser(description='The code for generating blurry images')

    parser.add_argument('--workers', default=8, type=int, help='number of workers for muti-thread processing')
    parser.add_argument('--data_root', type=str, help='the data root directory')
    parser.add_argument('--data_types', type=str, help='the data root directory')
    parser.add_argument('--left_right', type=str, help='the data root directory')
    parser.add_argument('--exposure_rate', type=float, help='the data root directory')

    return parser.parse_args()

def blur_one_trajectory(args, traj_dir):
    
    print('Deal with {}'.format(traj_dir))
    start_time = time.time()

    img_dir = os.path.join(traj_dir, args.left_right)
    reverse_flow_dir = os.path.join(traj_dir, 'flow_reverse')
    save_blur_img_dir = os.path.join(traj_dir, '{}_blur_{}'.format(args.left_right, args.exposure_rate))

    if not os.path.exists(save_blur_img_dir):
            os.makedirs(save_blur_img_dir)
    img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    img_ids = [img_path.split('/')[-1].split('_')[0] for img_path in img_paths]
    img_ids = sorted(img_ids)

    img_paths = [os.path.join(img_dir, img_id + '_left.png') for img_id in img_ids[1:]]
    flow_paths = [os.path.join(reverse_flow_dir, img_id2 + '_' + img_id1 + '_flow.npy')\
                for img_id1, img_id2 in zip(img_ids[:-1], img_ids[1:])]
    save_blur_img_dirs = [save_blur_img_dir] * len(flow_paths)
        
    data = list(zip(img_paths, flow_paths, save_blur_img_dirs))

    p = multiprocessing.Pool(args.workers)
    p.map(blur_image_multithread_wrapper, data)

    print('Deal with path {} takes {} secs'.format(traj_dir, time.time() - start_time))


if __name__ == '__main__':
    
    args = arg_parse()

    # path_data_roots = ['/data/datasets/wenshanw/tartan_data/soulcity/Data_fast/P006', 
    #                   '/data/datasets/wenshanw/tartan_data/hongkongalley/Data_fast/P005',
    #                   '/data/datasets/wenshanw/tartan_data/ocean/Data_fast/P007',
    #                   '/data/datasets/wenshanw/tartan_data/gascola/Data/P004']
    # test_on_predefined_image()
    # path_data_roots = ['/home/chaotec/chessboard/P000/',
    #                    '/home/chaotec/chessboard/P001/',
    #                    '/home/chaotec/chessboard/P002/',
    #                    '/home/chaotec/chessboard/P003/']
    
    data_types = args.data_types.split(',')
    
    for env_dir in os.listdir(args.data_root):
        
        env_dir = os.path.join(args.data_root, env_dir)
        
        if not os.path.isdir(env_dir): continue
        
        for data_type in data_types:
           
            data_dir = os.path.join(env_dir, data_type)
            
            if not os.path.exists(data_dir): break

            traj_dirs = glob.glob(os.path.join(data_dir, 'P*'))
            
            for traj_dir in traj_dirs:

                blur_one_trajectory(args, traj_dir)
