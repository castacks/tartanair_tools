import cv2
import numpy as np

def add_motion_blur(img, flow):
	pass


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

if __name__ == '__main__':
	# sample code 
	imgfile1 = 'data/000380_left.png'
	imgfile2 = 'data/000381_left.png'
	flowfile = 'data/000380_000381_flow.npy'

	img1 = load_rgb(imgfile1)
	img2 = load_rgb(imgfile2)
	flow = load_flow(flowfile)
	flowvis = visflow(flow)

	visimg = np.concatenate((img1,img2, flowvis), axis=0)
	visimg = cv2.resize(visimg, (0, 0), fx=0.5, fy=0.5)
	cv2.imshow('img', visimg)
	cv2.waitKey(0)
