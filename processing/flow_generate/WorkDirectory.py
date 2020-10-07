from __future__ import print_function

import json
import math
import numpy as np
import os

def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """

    qi2 = q[0, 0]**2
    qj2 = q[1, 0]**2
    qk2 = q[2, 0]**2

    qij = q[0, 0] * q[1, 0]
    qjk = q[1, 0] * q[2, 0]
    qki = q[2, 0] * q[0, 0]

    qri = q[3, 0] * q[0, 0]
    qrj = q[3, 0] * q[1, 0]
    qrk = q[3, 0] * q[2, 0]

    s = 1.0 / ( q[3, 0]**2 + qi2 + qj2 + qk2 )
    ss = 2 * s

    R = [\
        [ 1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj) ],\
        [ ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri) ],\
        [ ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2) ],\
    ]

    R = np.array(R, dtype = np.float64)

    return R

def get_pose_from_line(poseDataLine):
    """
    poseDataLine is a 7-element NumPy array. The first 3 elements are 
    the translations. The remaining 4 elements are the orientation 
    represented as a quternion.
    """

    data = poseDataLine.reshape((-1, 1))
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)

    return R.transpose(), -R.transpose().dot(t), q


def reshape_idx_array(idxArray):
    N = idxArray.size
    idxArray = idxArray.astype(np.int)

    # Find the closest squre root.
    s = int(math.ceil(math.sqrt(N)))

    # Create a new index array.
    idx2D = np.zeros( (s*s, ), dtype=np.int ) + idxArray.max()
    idx2D[:N] = idxArray

    # Reshape and transpose.
    idx2D = idx2D.reshape((s, s)).transpose()

    # Flatten again.
    return idx2D.reshape((-1, ))