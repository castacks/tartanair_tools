#!/usr/bin/python

from __future__ import print_function

import os
import cv2
import math
import time
import pandas
import multiprocessing

import numpy as np
import queue as queue # python3.
import numpy.linalg as LA
import WorkDirectory as WD

from os import listdir, mkdir
from Camera import CameraBase
from settings import get_args
from os.path import isfile, isdir

# NumPy dtype for PCL routines.
NP_FLOAT = np.float64
PCL_FLOAT = np.float32
TWO_PI = np.pi * 2

CROSS_OCC = 1
SELF_OCC  = 10
OUT_OF_FOV = 100

class FileLogger():
    def __init__(self, filename):
        if isfile(filename):
            timestr = time.strftime('%m%d_%H%M%S',time.localtime())
            filename = filename+'_'+timestr
        self.f = open(filename, 'w')

    def log(self, logstr):
        self.f.write(logstr)

    def logline(self, logstr):
        self.f.write(logstr+'\n')

    def close(self,):
        self.f.close()

def compose_m(R, t):
    m = np.zeros((4, 4), dtype=NP_FLOAT)
    m[0:3, 0:3] = R
    m[0:3, 3]   = t.reshape((-1,))
    m[3,3]      = 1.0

    return m

def inv_m(m):
    im = np.zeros_like(m, dtype=NP_FLOAT)

    R = m[0:3, 0:3]
    t = m[0:3, 3].reshape((3, 1))

    im[0:3, 0:3] = R.transpose()
    im[0:3, 3]   = -R.transpose().dot( t ).reshape((-1,))
    im[3, 3]     = 1.0

    return im

def du_dv(nu, nv, imageSize):
    wIdx = np.linspace( 0, imageSize[1] - 1, imageSize[1], dtype=np.int )
    hIdx = np.linspace( 0, imageSize[0] - 1, imageSize[0], dtype=np.int )

    u, v = np.meshgrid(wIdx, hIdx)

    return nu - u, nv - v

def map_normalized_angle(ang, maxVal=255):
    m = ang * maxVal * 2

    mask = m > maxVal
    m[mask] = maxVal*2 - m[mask]

    return np.clip( m, 0, maxVal )

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

def is_valid_ind(h,w,H,W):
    '''
    h, w: could be integer or float
    H, W: image height and width, should be integer
    return: whether (h,w) located in the image 
    '''
    h, w = round(h), round(w)
    if h>=0 and h<H and w>=0 and w<W:
        return True
    return False

def bilinear_interpolate(img, h, w):
    assert round(h)>=0 and round(h)<img.shape[0]
    assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

class GridMappingList(object):
    '''
    Hashmap from grid-index (i,j) to a list of values [v1,v2,..]
    In the warping case, vi: (warp_xi, warp_yi, i0, j0)
    '''
    def __init__(self, grid_x_size, grid_y_size):
        self.gridmap = {}

    def pts_within_thresh(self, h, w, thresh):
        adjPtInds = [(0,0),(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        roundH, roundW = int(round(h)), int(round(w))
        pts = []
        for deltaPt in adjPtInds: 
            adjPt = (roundH + deltaPt[0], roundW + deltaPt[1])
            if adjPt in self.gridmap:
                ptvalues = self.gridmap[adjPt]
                for ptvalue in ptvalues:
                    hh, ww, roundhh0, roundww0 = ptvalue
                    dd = math.sqrt( (hh-h)**2 + (ww-w)**2 )
                    if dd < thresh:
                        pt = (hh, ww, roundhh0, roundww0)
                        pts.append(pt)
        return pts

    def insert(self, th, tw, h0, w0):
        key = (int(round(th)), int(round(tw)))
        if key in self.gridmap:
            self.gridmap[key].append((th, tw, h0, w0))
        else:
            self.gridmap[key]= [(th, tw, h0, w0)]

def check_neighbor_dist(distArr, h, w, dist0): 
    '''
    To overcome the depth issue
    The warping process will have error on the edge of object 
    '''
    # adjPtInds = [(0,0),(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adjPtInds = [(0,0),(-1,0),(0,-1),(0,1),(1,0)]
    roundH, roundW = int(round(h)), int(round(w))
    H, W = distArr.shape[0], distArr.shape[1]
    mindist = 100000
    mindist_sign = 100000
    for deltaPt in adjPtInds: 
        hh, ww = roundH + deltaPt[0], roundW + deltaPt[1]
        if is_valid_ind(hh,ww,H,W):
            dist1 = distArr[roundH + deltaPt[0], roundW + deltaPt[1]]
            dist =  dist0 - dist1 
            if abs(dist) < mindist:
                mindist = abs(dist)
                mindist_sign = dist 
    # if mindist_sign < -100:
    #     print('dist<0, {}'.format(mindist))
    #     print(distArr[roundH-1:roundH+2, roundW-1:roundW+2])
    #     import ipdb;ipdb.set_trace()
    return mindist_sign

def warp_10(img1, u, v, X_01C_dist, X1C_dist, z, occThresh = 0.707, occDistThresh = 0.5):
    '''
    warpMap: (h_on_f1_round, w_on_f1_round) -> [(h_on_f1, w_on_f1, h_on_f0_round, w_on_f0_round), ...]
    h_on_f1, w_on_f1 are used to find closest points
    h_on_f0_round, w_on_f0_round are used to update the map if occlusion happens
    occThresh: if two pixels on img0 is warped to two locations on img1 that as close as occThresh
    occDistThresh: if the distance of warped point is different with the point on img1 above this occDistThresh
    '''
    # warpMap = {} # map: 
    ImageH = img1.shape[0]
    ImageW = img1.shape[1]
    warpMap = GridMappingList(ImageH, ImageW)
    maskOutofView = np.zeros((ImageH, ImageW),dtype=np.bool)
    maskOcclusion = np.zeros((ImageH, ImageW),dtype=np.bool)
    maskOccbyOut = np.zeros((ImageH, ImageW),dtype=np.bool)
    warpImg = np.zeros((ImageH, ImageW, 3),dtype=np.uint8)
    for h in range(ImageH):
        for w in range(ImageW):
            targetH = v[h,w]
            targetW = u[h,w]
            if is_valid_ind(targetH, targetW, ImageH, ImageW):
                warpImg[h, w, :] = bilinear_interpolate(img1, targetH, targetW)
                # calculate occlusion mask
                pts = warpMap.pts_within_thresh(targetH, targetW, occThresh)
                dist0 = X_01C_dist[h*ImageW+w] # get_distance_from_coordinate_table(X_01C, h*ImageW+w)
                dist1 = check_neighbor_dist(X1C_dist.reshape(ImageH, ImageW), targetH, targetW, dist0) #bilinear_interpolate(X1C_dist.reshape(ImageH, ImageW, 1), targetH, targetW) # X1C_dist[int(round(targetH)*ImageW + round(targetW))] # get_distance_from_coordinate_table(X1C, int(round(targetH)*ImageW + round(targetW)))
                if abs(dist1) > occDistThresh:
                    maskOccbyOut[h, w] = True
                if len(pts)>0: # found close points, occlusion happens, compare the distance
                    # import ipdb;ipdb.set_trace()
                    occlude = False
                    for pt in pts: 
                        pt_h0, pt_w0 = pt[2], pt[3]
                        if abs(pt_h0-h)<=1 and abs(pt_w0-w)<=1: # adjacent points can not occlude each other
                            continue
                        dist_pt = X_01C_dist[pt_h0*ImageW+pt_w0] # get_distance_from_coordinate_table(X_01C, pt_h0*ImageW+pt_w0)
                        if dist_pt < dist0:
                            maskOcclusion[h, w] = True
                            occlude = True
                            break
                    if not occlude: # (h, w) is a closest point
                        for pt in pts:
                            pt_h0, pt_w0 = pt[2], pt[3]
                            if abs(pt_h0-h)<=1 and abs(pt_w0-w)<=1: # adjacent points can not occlude each other
                                continue
                            maskOcclusion[pt_h0, pt_w0] = True

                warpMap.insert(targetH, targetW, h, w)
            else:
                maskOutofView[h,w] = True
    # import ipdb;ipdb.set_trace()
    maskOutofView[z<0] = True
    return warpImg, maskOcclusion, maskOutofView, maskOccbyOut

def load_pose_id_pose_data_from_folder(poseDataFn, imgdir):
    """
    inputs:
        poseDataFn: the path to the pose file
                    e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000/pose_left.txt
        imgdir:     the path to the image sequence directory (the same as trajpath)
                    e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000/image_left
    outputs:
        poseIDs: the list of image ids 
                    e.g. [000000, 000001, ... 000845.... ]
        poseData: an numpy array with shape (n, 7), where n is the number of images 
    """

    imgfiles = os.listdir(imgdir)
    poseIDs = [fn.split('_')[0] for fn in imgfiles if fn[-4:]=='.png']
    poseIDs.sort()

    if ( ".txt" == os.path.splitext( os.path.split(poseDataFn)[1] )[1] ):
        poseData = np.loadtxt( poseDataFn, dtype=NP_FLOAT )
    else:
        poseData = np.load( poseDataFn ).astype(NP_FLOAT)

    assert len(poseIDs) == len(poseData)
    
    return poseIDs, poseData


def save_flow(fnBase, flowSuffix, maskSuffix, du, dv, maskFOV=None, maskOcc=None, maskOcc2=None):
    """
    fnBase: The file name base.
    flowSuffix: Filename suffix of the optical flow file.
    maskSuffix: Filename suffix of the mask file.
    du, dv: The opitcal flow saved as NumPy array with dtype=numpy.float32.
    maskFOV: The FOV mask. NumPy array with dtype=numpy.bool.
    maskOcc: The occlusion mask. NumPy array with dtype=numpy.bool.
    maskOcc2: The occlusion mask. NumPy array with dtype=numpy.bool.
    """

    # Create a 2-channel NumPy array.
    flow = np.stack([du, dv], axis=-1).astype(np.float32)

    # Create a 1-channel NumPy array.
    mask = np.zeros_like(du, dtype=np.uint8)

    if ( maskOcc2 is not None ):
        mask[maskOcc2] = CROSS_OCC

    if ( maskOcc is not None ):
        mask[maskOcc] = SELF_OCC

    if ( maskFOV is not None ):
        mask[maskFOV] = OUT_OF_FOV

    np.save( "%s%s.npy" % (fnBase, flowSuffix), flow )
    np.save( "%s%s.npy" % (fnBase, maskSuffix), mask )

def coordinate2distance(cordinates):
    '''
    cordinates: N x 3 np array
    return: N x 1 np array
    '''
    return np.sqrt(cordinates[0,:]*cordinates[0,:] + cordinates[1,:]*cordinates[1,:] + cordinates[2,:]*cordinates[2,:])

def save_debug_image(img0, img1, warpImg, diff, maskOcclusion, maskOutofView, maskOccbyOut, maskcombine, 
                        outDir, poseID_0, poseID_1, du, dv, scale=0.5):
    diffdisp =  diff.astype(np.uint8)
    disp0 = np.concatenate((img0, img1, warpImg, diffdisp), axis=1)
    # import ipdb;ipdb.set_trace()

    warpImgMask1 = np.zeros_like(img0, dtype=np.uint8)
    warpImgMask1[maskOcclusion] = [255,255,255]
    warpImgMask2 = np.zeros_like(img0, dtype=np.uint8)
    warpImgMask2[maskOutofView] = [255,255,255]
    warpImgMask3 = np.zeros_like(img0, dtype=np.uint8)
    warpImgMask3[maskOccbyOut] = [255,255,255]

    flow = np.stack([du, dv], axis=-1).astype(NP_FLOAT)
    flowdisp = visflow(flow)

    diffmaskdisp = diffdisp.copy()
    diffmaskdisp[maskcombine] = [0,0,0]
    maskdisp1 = np.zeros_like(img0, dtype=np.uint8)
    maskdisp1[maskOcclusion] = [255,255,255]
    maskdisp1[maskOutofView] = [255,255,255]
    maskdisp2 = np.zeros_like(img0, dtype=np.uint8)
    maskdisp2[maskOccbyOut] = [255,255,255]
    disp1 = np.concatenate((flowdisp, maskdisp1, maskdisp2, diffmaskdisp), axis=1)

    disp = np.concatenate((disp0, disp1), axis=0)

    # cv2.imshow('img', disp)
    # cv2.waitKey(0)
    disp = cv2.resize(disp, (0,0), fx=scale, fy=scale)
    cv2.imwrite(outDir+'/warp_'+poseID_0+'_'+poseID_1+'.jpg', disp)
    print("  image saved {}".format(outDir+'/warp_'+poseID_0+'.jpg'))


def process_single_process(name, outDir, \
    imgDir, poseID_0, poseID_1, poseDataLine_0, poseDataLine_1, depth_0, depth_1, cam, 
    imgSuffix='_left', imgExt='.png', distanceRange=1000, save_flow_image=True):
    
    fnBase = "%s/%s_%s" % (outDir, poseID_0, poseID_1)
    save_flow_path = "%s%s.npy" % (fnBase, "_flow")
    save_mask_path = "%s%s.npy" % (fnBase, "_mask")

    if os.path.exists(save_flow_path) and os.path.exists(save_mask_path):
        print('save_flow_path, {}, has been created'.format(save_flow_path))
        return 0, 0, 0

    # Get the pose of the first position.
    R0, t0, q0 = WD.get_pose_from_line(poseDataLine_0)
    R0Inv = LA.inv(R0)
    M0 = compose_m(R0, t0)
    
    # Get the pose of the second position.
    R1, t1, q1 = WD.get_pose_from_line(poseDataLine_1)
    R1Inv = LA.inv(R1)
    M1 = compose_m(R1, t1)
    
    # Compute the rotation between the two camera poses.
    R = np.matmul( R1, R0Inv )

    # Calculate the coordinates in the first camera's frame.
    X0C = cam.from_depth_to_x_y(depth_0) # Coordinates in the camera frame. z-axis pointing forwards.
    X0  = cam.worldRI.dot(X0C)           # Corrdinates in the NED frame. z-axis pointing downwards.

    # The coordinates in the world frame.
    XWorld_0  = R0Inv.dot(X0 - t0)

    # Calculate the coordinates in the second camera's frame.
    X1C = cam.from_depth_to_x_y(depth_1) # Coordinates in the camera frame. z-axis pointing forwards.
    X1C_dist = coordinate2distance(X1C)
    X1C_dist = np.clip(X1C_dist, 0, distanceRange) # Depth clipping.

    # ====================================
    # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
    X_01 = R1.dot(XWorld_0) + t1

    # The image coordinates in the second camera.
    X_01C = cam.worldR.dot(X_01)                  # Camera frame, z-axis pointing forwards.
    X_01C_dist = coordinate2distance(X_01C)
    X_01C_dist = np.clip(X_01C_dist, 0, distanceRange) # Depth clipping.

    c     = cam.from_camera_frame_to_image(X_01C) # Image plane coordinates.
    z = X_01C[2, :].reshape(cam.imageSize) # verify whether points go behind the camera
    # print('behind camera: {}'.format(np.sum(z<0.)))

    # Get new u anv v
    u = c[0, :].reshape(cam.imageSize)
    v = c[1, :].reshape(cam.imageSize)

    # Get the du and dv.
    du, dv = du_dv(u, v, cam.imageSize)

    # read image 
    starttime = time.time()
    cam0ImgFn = "%s/%s%s%s" % ( imgDir, poseID_0, imgSuffix, imgExt )
    cam1ImgFn = "%s/%s%s%s" % ( imgDir, poseID_1, imgSuffix, imgExt )
    img0 = cv2.imread( cam0ImgFn, cv2.IMREAD_UNCHANGED )
    img1 = cv2.imread( cam1ImgFn, cv2.IMREAD_UNCHANGED )
    warpImg, maskOcclusion, maskOutofView, maskOccbyOut = warp_10(img1, u, v, X_01C_dist, X1C_dist, z)
    print("  warping time: {}".format(time.time()-starttime))

    # combine three mask into one
    validCombine = np.ones_like(maskOcclusion, dtype=np.bool)
    validCombine[maskOcclusion] = False
    validCombine[maskOccbyOut] = False
    validCombine[maskOutofView] = False
    maskcombine = (1 - validCombine).astype(np.bool)
    invalidnum = np.sum(maskcombine)

    diff = np.abs(warpImg.astype(np.float32) - img0)
    diffmask = diff.copy()
    diffmask[maskcombine] = 0
    maskedError = diff[validCombine].mean()
    
    # import ipdb;
    # ipdb.set_trace()

    print("  error: {}, {}, {}".format(diff.mean(), maskedError, invalidnum))

    save_flow( "%s/%s_%s" % (outDir, poseID_0, poseID_1), "_flow", "_mask", du, dv, maskOutofView, maskOcclusion, maskOccbyOut)
    print("  Flow saved.")
    if save_flow_image:
        save_debug_image(img0, img1, warpImg, diff, maskOcclusion, maskOutofView, maskOccbyOut, maskcombine, 
            outDir, poseID_0, poseID_1, du, dv)

    return warpImg, maskedError, invalidnum

def logging_worker(name, jq, p, workingDir):
    
    import logging

    logger = logging.getLogger("ImageFlow")
    logger.setLevel(logging.INFO)

    logFn = "%s/flow_log_ctchou.log" % (workingDir)
    print(logFn)

    fh = logging.FileHandler( logFn, "w" )
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("%s: Logger initialized." % (name))

    while (True):
        if (p.poll()):
            command = p.recv()

            print("%s: %s command received." % (name, command))

            if ("exit" == command):
                # print("%s: exit." % (name))
                break
        
        try:
            job = jq.get(False)
            # print("{}: {}.".format(name, jobStrList))

            logger.info(job)

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    logger.info("Logger exited.")

def worker(name, jq, rq, lq, p, args, trajpath, flowpath):
    """
    name: String, the name of this worker process.
    jq: A JoinableQueue. The job queue.
    rq: The report queue.
    lq: The logger queue.
    p: A pipe connection object. Only for receiving.
    """

    lq.put("%s: Worker starts." % (name))

    # ==================== Preparation. ========================
    # Camera.
    cam = CameraBase(args.focal, [args.image_height, args.image_width])

    depthDir      = trajpath + "/" + 'depth_left'
    imgDir        = trajpath + "/" + 'image_left'
    depthTail     = '_left_depth.npy' #inputParams["depthSuffix"] + inputParams["depthExt"]

    count = 0

    while (True):
        if (p.poll()):
            command = p.recv()

            lq.put("%s: %s command received." % (name, command))

            if ("exit" == command):
                # print("%s: exit." % (name))
                break

        try:
            job = jq.get(True, 1)
            # print("{}: {}.".format(name, jobStrList))

            poseID_0 = job["poseID_0"]
            poseID_1 = job["poseID_1"]
            poseDataLineList_0 = job["poseLineList_0"]
            poseDataLineList_1 = job["poseLineList_1"]

            poseDataLine_0 = np.array( poseDataLineList_0, dtype=NP_FLOAT )
            poseDataLine_1 = np.array( poseDataLineList_1, dtype=NP_FLOAT )

            # Load the depth.
            depth_0 = np.load( depthDir + "/" + poseID_0 + depthTail ).astype(NP_FLOAT)
            depth_1 = np.load( depthDir + "/" + poseID_1 + depthTail ).astype(NP_FLOAT)

            # Process.
            warppedImg, maskedError, invalidnum = \
                process_single_process(name, flowpath, 
                    imgDir, poseID_0, poseID_1, poseDataLine_0, poseDataLine_1, 
                    depth_0, depth_1, cam, save_flow_image=args.save_flow_image)

            rq.put( { "idx": job["idx"], \
                "poseID_0": poseID_0, "poseID_1": poseID_1, 
                "error": maskedError, "invalid_num": invalidnum } )

            count += 1

            lq.put("%s: idx = %d. " % (name, job["idx"]))

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    lq.put("%s: Done with %d jobs." % (name, count))


def process_report_queue(rq, jobnum):
    """
    rq: A Queue object containing the report data.
    """
    
    count = 0

    mergedDict = {"idx": [], 
                  "poseID_0": [], 
                  "poseID_1": [],
                  "error": [], 
                  "invalid_num": [] 
                 }

    while (count < jobnum):
        try:
            r = rq.get(block=False)

            mergedDict['idx'].append( r["idx"] )
            mergedDict['poseID_0'].append( r["poseID_0"] )
            mergedDict['poseID_1'].append( r["poseID_1"] )
            mergedDict['error'].append( r["error"] )
            mergedDict['invalid_num'].append( r["invalid_num"] )

            count += 1
        except queue.Empty as ex:
            time.sleep(1) 

    return mergedDict

def save_report(fn, report):
    """
    fn: the output filename.
    report: a dictonary contains the data.

    This function will use pandas package to output the report as a CSV file.
    """

    # Create the DataFrame object.
    df = pandas.DataFrame.from_dict(report, orient="columns")

    # Sort the rows according to the index column.
    df.sort_values(by=["idx"], ascending=True, inplace=True)

    # Save the file.
    df.to_csv(fn, index=False)


def process_trajectory(args, imgpath, posefile, flowpath, trajpath):
    """
    inputs:
        args: arguments and parameters
        imgpath: the path to the folder of images. 
                 e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000/image_left
        posefile: the path to a text pose file.
                 e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000/pose_left.txt
        flowpath: the path to the directory to save flow images
                 e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000/flow2_1007_140542
        trajpath: the path to the folder of a trajectory, a sequence of images
                 e.g. /data/datasets/wenshanw/tartan_data/oldtown/Data/P000
    """

    # Load the pose filenames and the pose data.
    # poseIDs, poseData = load_pose_id_pose_data( inputParams, args )
    poseIDs, poseData = load_pose_id_pose_data_from_folder(posefile, imgpath)
    print("poseData and poseFilenames loaded.")
    
    # Get the number of poseIDs.
    nPoses = len( poseIDs )
    idxNumberRequest = nPoses #inputParams["idxNumberRequest"]

    idxStep = args.index_step # inputParams["idxStep"]
    
    if idxStep > 0:
        
        idxList = list(range(args.start_index, nPoses - idxStep)) 
                # [ i for i in range(args.start_index, nPoses - idxStep) ]
        if ( idxNumberRequest < len(idxList)-1 ):
            idxList = idxList[:idxNumberRequest+1]
        idxArray = np.array(idxList, dtype=np.int)
         
    elif idxStep < 0: # for gereating flow from t+1 -> t
        
        idxList = list(range(args.start_index - idxStep, nPoses))
        idxArray = np.array(idxList, dtype=np.int)[::-1]

    # # Reshape the idxArray.
    # idxArrayR = WD.reshape_idx_array(idxArray)

    startTime = time.time()

    print("Main: Main process.")

    # Process.
    jqueue  = multiprocessing.JoinableQueue() # The job queue.
    manager = multiprocessing.Manager()
    rqueue  = manager.Queue()         # The report queue.

    loggerQueue = multiprocessing.JoinableQueue()
    [conn1, loggerPipe] = multiprocessing.Pipe(False)

    loggerProcess = multiprocessing.Process( \
        target=logging_worker, args=["Logger", loggerQueue, conn1, trajpath] )

    loggerProcess.start()

    processes   = []
    processStat = []
    pipes       = []

    loggerQueue.put("Main: Create %d processes." % (args.np))

    for i in range(args.np):
        [conn1, conn2] = multiprocessing.Pipe(False)
        processes.append( multiprocessing.Process( \
            target=worker, args=["P%03d" % (i), jqueue, rqueue, loggerQueue, conn1, args, trajpath, flowpath]) )
        pipes.append(conn2)

        processStat.append(1)

    for p in processes:
        p.start()

    loggerQueue.put("Main: All processes started.")
    loggerQueue.join()

    nIdx   = idxArray.size
    
    for i in range(nIdx):
        
        # The index of cam_0.
        idx_0 = idxArray[i]

        idx_1 = idx_0 + idxStep

        # Get the poseIDs.
        poseID_0 = poseIDs[ idx_0 ]
        poseID_1 = poseIDs[ idx_1 ]
        
        # Get the poseDataLines.
        poseDataLine_0 = poseData[idx_0].reshape((-1, )).tolist()
        poseDataLine_1 = poseData[idx_1].reshape((-1, )).tolist()

        d = { "idx": idx_0, "poseID_0": poseID_0, "poseID_1": poseID_1, \
            "poseLineList_0": poseDataLine_0, "poseLineList_1": poseDataLine_1 }

        jqueue.put(d)
        
    loggerQueue.put("Main: All jobs submitted.")

    # Process the rqueue.
    report = process_report_queue(rqueue, nIdx-1)

    jqueue.join()

    loggerQueue.put("Main: Job queue joined.")
    loggerQueue.join()

    # Save the report to file.
    reportFn = "%s/Report.csv" % (flowpath)
    save_report(reportFn, report)
    loggerQueue.put("Report saved to %s. " % (reportFn))

    endTime = time.time()

    loggerQueue.put("%d poses, starting at idx = %d, step = %d, %d steps in total. idxNumberRequest = %d. Total time %ds. \n" % \
        (nPoses, args.start_index, idxStep, len( idxList )-1, idxNumberRequest, endTime-startTime))

    # Stop all subprocesses.
    for p in pipes:
        p.send("exit")

    loggerQueue.put("Main: Exit command sent to all processes.")

    loggerQueue.join()

    nps = len(processStat)

    for i in range(nps):
        p = processes[i]

        if ( p.is_alive() ):
            p.join() # timeout=1

        if ( p.is_alive() ):
            loggerQueue.put("Main: %d subprocess (pid %d) join timeout. Try to terminate" % (i, p.pid))
            p.terminate()
        else:
            processStat[i] = 0
            loggerQueue.put("Main: %d subprocess (pid %d) joined." % (i, p.pid))

    if (not 1 in processStat ):
        loggerQueue.put("Main: All processes joined. ")
    else:
        loggerQueue.put("Main: Some process is forced to be terminated. ")

    # Stop the logger.
    loggerQueue.join()
    loggerPipe.send("exit")
    loggerProcess.join()

    print("Main: Done.")

# python flow_and_warping_error.py --data-root '' --data-folders '' -- env-folders '' --index-step 2 --flow-outdir flow2 --np 8
if __name__ == "__main__":
    
    # Script input arguments.
    args = get_args()

    data_root_dir = args.data_root
    data_folders = args.data_folders.split(',')
    
    # find env folders
    if args.env_folders=='': # read all available folders in the data_root_dir
        env_folders = listdir(data_root_dir)    
    else:
        env_folders = args.env_folders.split(',')
    print('Detected envs {}'.format(env_folders))

    logf = FileLogger( os.path.join(data_root_dir, 'flow_error.log') )

    for env_folder in env_folders:

        env_dir = os.path.join(data_root_dir, env_folder)
        print('Working on env {}'.format(env_dir))

        for data_folder in data_folders:
            
            datapath = os.path.join(env_dir, data_folder)
            
            if not isdir(datapath):
                
                logf.logline('Data folder missing ' + datapath)
                print('!!data folder missing '+ datapath)
                
                continue
            
            print('    Opened data folder {}'.format(datapath))
 
            trajfolders = listdir(datapath)
            trajfolders = [ tf for tf in trajfolders if tf[0]=='P' ]
            trajfolders.sort()
            print('    Found {} trajectories'.format(len(trajfolders)))

            for trajfolder in trajfolders:

                trajpath = os.path.join(datapath, trajfolder) # datapath +'/' + trajfolder
                
                # flow path is the path to store the generated flow images
                flowpath = os.path.join(trajpath, args.flow_outdir) # trajpath +'/' + args.flow_outdir
                
                if not isdir(flowpath):
                    
                    mkdir(flowpath)
                
                elif not args.force_overwrite:
                    
                    timestr  = time.strftime('%m%d_%H%M%S',time.localtime())
                    flowpath = flowpath + '_' + timestr
                    mkdir(flowpath)
                    logf.logline('Flow folder exists, create new ' + flowpath)
                    print('Flow folder exists, create new ' + flowpath)

                imgpath  = os.path.join(trajpath, 'image_left') # trajpath +'/image_left'
                posefile = os.path.join(trajpath, 'pose_left.txt') # trajpath +'/pose_left.txt'

                if not isdir(imgpath):
                    
                    logf.logline('Can not find image folder ' + imgpath)
                    print('    !!image folder missing ' + imgpath)
                    
                    continue
                
                if not isfile(posefile):
                    
                    logf.logline('Can not find pose file ' + posefile)
                    print('    !!pose file missing ' + posefile)
                    
                    continue
                
                process_trajectory(args, imgpath, posefile, flowpath, trajpath)
                
                # break # for debugging
        
        # break # for debugging
