

### Camera intrinsics 
```
fx = 320.0  # focal length x
fy = 320.0  # focal length y
cx = 320.0  # optical center x
cy = 240.0  # optical center y

fov = 90 deg # field of view

width = 640
height = 320
```

### Depth image

### Segmentation image

### Optical flow

### Pose file

The camera pose file is a text file containing the translation and orientation of the camera in a fixed coordinate frame. Note that also our automatic evaluation tool expects both the ground truth trajectory and the estimated trajectory to be in this format. 

* Each line in the text file contains a single pose.

* The number of lines/poses must be the same as the number of image frames in that trajectory. 

* The format of each line is '**tx ty tz qx qy qz qw**'. 

* **tx ty tz** (3 floats) give the position of the optical center of the color camera with respect to the world origin in the world frame.

* **qx qy qz qw** (4 floats) give the orientation of the optical center of the color camera in the form of a unit quaternion with respect to the world frame. 

* The camera motion is defined in the NED frame. That is to say, the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right, the z-axis is pointing to the camera's downward.