from __future__ import print_function

import copy
import numpy as np

NP_FLOAT = np.float64

def clip_depth(depth, dMax):
    if ( dMax > depth.min() ):
         return np.clip( depth, 0, dMax )
    else:
        raise Exception("dMax = {} is not appropirate. depth.min() = {}, depth.max() = {}. ".format( \
            dMax, depth.min(), depth.max() ))

class CameraBase(object):
    def __init__(self, focal, imageSize):
        self.focal = focal
        self.imageSize = copy.deepcopy(imageSize) # List or tuple, (height, width)
        self.size = self.imageSize[0] * self.imageSize[1]

        self.pu = self.imageSize[1] / 2
        self.pv = self.imageSize[0] / 2

        self.cameraMatrix = np.eye(3, dtype = NP_FLOAT)
        self.cameraMatrix[0, 0] = self.focal
        self.cameraMatrix[1, 1] = self.focal
        self.cameraMatrix[0, 2] = self.pu
        self.cameraMatrix[1, 2] = self.pv

        self.worldR = np.zeros((3,3), dtype = NP_FLOAT)
        self.worldR[0, 1] = 1.0
        self.worldR[1, 2] = 1.0
        self.worldR[2, 0] = 1.0

        self.worldRI = np.zeros((3,3), dtype = NP_FLOAT)
        self.worldRI[0, 2] = 1.0
        self.worldRI[1, 0] = 1.0
        self.worldRI[2, 1] = 1.0

    def from_camera_frame_to_image(self, coor):
        """
        coor: A numpy column vector, 3x1.
        return: A numpy column vector, 2x1.
        """
        
        # coor = self.worldR.dot(coor)
        x = self.cameraMatrix.dot(coor)
        x = x / x[2,:]

        return x[0:2, :]

    def from_depth_to_x_y(self, depth, dMax=None):
        wIdx = np.linspace( 0, self.imageSize[1] - 1, self.imageSize[1], dtype=np.int )
        hIdx = np.linspace( 0, self.imageSize[0] - 1, self.imageSize[0], dtype=np.int )

        u, v = np.meshgrid(wIdx, hIdx)

        u = u.astype(NP_FLOAT)
        v = v.astype(NP_FLOAT)
        
        if ( dMax is not None):
            depth = clip_depth( depth, dMax )

        x = ( u - self.pu ) * depth / self.focal
        y = ( v - self.pv ) * depth / self.focal

        coor = np.zeros((3, self.size), dtype = NP_FLOAT)
        coor[0, :] = x.reshape((1, -1))
        coor[1, :] = y.reshape((1, -1))
        coor[2, :] = depth.reshape((1, -1))

        # coor = self.worldRI.dot(coor)

        return coor
