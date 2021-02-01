import os
import cv2

import numpy as np

from motion_blur import visflow, load_rgb, flow16to32

def visualize_flow_image(img1_path, img2_path, flow_path):

    img1 = load_rgb(img1_path)
    img2 = load_rgb(img2_path)
    flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
    
    print('flow range: np.amax(flow): {}, np.amin(flow): {}'.format(np.amax(flow), np.amin(flow)))
    flow, _ = flow16to32(flow)

    flowvis = visflow(flow)
    
    visimg = np.concatenate((img1, img2, flowvis), axis=0)
    visimg = cv2.resize(visimg, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('./img_collage.png', visimg)

if __name__ == '__main__':
    
    data_dir = '/data/datasets/wenshanw/tartan_data/oldtown/Data/P000'
    
    img1_path = os.path.join(data_dir, 'image_right', '001690_right.png') #'001675_001674_flow.png')
    img2_path = os.path.join(data_dir, 'image_right', '001689_right.png')
    flow_path = os.path.join(data_dir, 'flow_reverse_right', '001690_001689_flow.png')

    visualize_flow_image(img1_path, img2_path, flow_path)
