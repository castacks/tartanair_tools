import os
import cv2
import glob

from settings import get_args
from flow_and_warping_error import process_trajectory


def check_and_make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':

    # data_dir = '/home/chaotec/chessboard/'
        
    args = get_args()
    data_dir = args.data_root

    traj_paths = glob.glob(os.path.join(data_dir, 'P00*'))

    
    for idx, traj_path in enumerate(traj_paths):

        img_dir = os.path.join(traj_path, 'image_left')
        pose_file = os.path.join(traj_path, 'pose_left.txt')
        flow_save_dir = os.path.join(traj_path, 'flow_reverse')
        print('==========> Dea with trajectory {}'.format(idx)) 
        print('traj_path: {}'.format(traj_path))
        print('pose_file: {}'.format(pose_file))
        print('img_dir: {}'.format(img_dir))
        print('flow_save_dir: {}'.format(flow_save_dir))
        check_and_make_dir(flow_save_dir)
        process_trajectory(args, img_dir, pose_file, flow_save_dir, traj_path)
