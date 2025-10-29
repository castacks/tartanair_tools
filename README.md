🔥 The TartanAir-V2 is released! Please checkout the [TartanAir-V2 website](tartanair.org)!

The dataset is available from both Hugging Face and the Airlab server! 
If anyone or any research group is interested in hosting our dataset, please contact wenshanw@andrew.cmu.edu. 

# TartanAir dataset: AirSim Simulation Dataset for Simultaneous Localization and Mapping
This repository provides sample code and scripts for accessing the training and testing data, as well as evaluation tools. Please refer to [TartanAir](http://theairlab.org/tartanair-dataset) for more information about the dataset. 
You can also reach out to contributors on the associated [AirSim GitHub](https://github.com/microsoft/AirSim).

This dataset was used to train the first generalizable learning-based visual odometry [TartanVO](http://theairlab.org/tartanvo/), which achieved better performance than geometry-based VO methods in challenging cases. Please check out the [paper](https://arxiv.org/pdf/2011.00359.pdf) and published [TartanVO code](https://github.com/castacks/tartanvo). 

## Download the training data

The data is divided into two levels (Easy and Hard) in terms of the motion patterns. It is organized in trajectory folders. You can download data from different cameras (left or right), with different data types (RGB, depth, segmentation, camera pose, and flow). Please see [data type](data_type.md) page for the camera intrinsics, extrinsics and other information. 


<p style="color:red"> <b> !! NOTE: The size of all the data is up to 3TB! It could take days to download. We also added the option to use the dataset directly on Azure without requiring a download. Please select the data type you really need before download. You can also go to 
<a href=http://theairlab.org/tartanair-dataset>TartanAir</a> 
to download the sample files for a better understanding of the data types. </b> </p>

###  Data directory structure

```
ROOT
|
--- ENV_NAME_0                             # environment folder
|       |
|       ---- Easy                          # difficulty level
|       |      |
|       |      ---- P000                   # trajectory folder
|       |      |      |
|       |      |      +--- depth_left      # 000000_left_depth.npy - 000xxx_left_depth.npy
|       |      |      +--- depth_right     # 000000_right_depth.npy - 000xxx_right_depth.npy
|       |      |      +--- flow            # 000000_000001_flow/mask.npy - 000xxx_000xxx_flow/mask.npy
|       |      |      +--- image_left      # 000000_left.png - 000xxx_left.png 
|       |      |      +--- image_right     # 000000_right.png - 000xxx_right.png 
|       |      |      +--- seg_left        # 000000_left_seg.npy - 000xxx_left_seg.npy
|       |      |      +--- seg_right       # 000000_right_seg.npy - 000xxx_right_seg.npy
|       |      |      ---- pose_left.txt 
|       |      |      ---- pose_right.txt
|       |      |  
|       |      +--- P001
|       |      .
|       |      .
|       |      |
|       |      +--- P00K
|       |
|       +--- Hard
|
+-- ENV_NAME_1
.
.
|
+-- ENV_NAME_N
```

### Download data to your local machine

We provide a python script `download_training.py` for the data downloading. You can also take a look at the [URL list](download_training_zipfiles.txt) to download the specific files you want. 

* Install dependencies

  `pip install boto3 colorama minio`

* Specify an output directory

  --output-dir OUTPUTDIR

* Select file type:

  --rgb

  --depth

  --seg

  --flow

* Select difficulty level:
  
  --only-hard

  --only-easy

  [NO TAG]: both 'hard' and 'easy' levels

* Select camera:
  
  --only-left

  --only-right

  [NO TAG]: both 'left' and 'right' cameras

* Select flow type when --flow is set:
  
  --only-flow

  --only-mask

  [NO TAG]: both 'flow' and 'mask' files

* Unzip the files after downloading: 

  --unzip

For example, download all the RGB images from the left camera:

```
python download_training.py --output-dir OUTPUTDIR --rgb --only-left --unzip
```

Download all the depth data from both cameras in hard level: 

```
python download_training.py --output-dir OUTPUTDIR --depth --only-hard --unzip
```

Download all optical flow data without flow-mask:

```
python download_training.py --output-dir OUTPUTDIR --flow --only-flow --unzip
```

Download all the files in the dataset (could be very slow due to the large size):

```
python download_training.py --output-dir OUTPUTDIR --rgb --depth --seg --flow --unzip
```

Our data is hosted on two servers located in the United States. By default, it downloads from [AirLab](https://theairlab.org/) data server. If you encounter any network issues, please try adding `--huggingface` for an alternative source. 

---


## Download the testing data for the CVPR Visual SLAM challenge

* [Monocular track](https://drive.google.com/file/d/1N9BkpQuibIyIBkLxVPUuoB-eDOMFqY8D/view?usp=sharing) (Size: 7.65 GB)
  
  MD5 hash: 009b52e7d7b224ffb8a203db294ac9fb

```
mono
|
--- ME000                             # monocular easy trajectory 0 
|       |
|       ---- 000000.png               # RGB image 000000
|       ---- 000001.png               # RGB image 000001
|       .
|       .
|       ---- 000xxx.png               # RGB image 000xxx
|
+-- ME001                             # monocular easy trajectory 1 
.
.
+-- ME007                             # monocular easy trajectory 7 
|
+-- MH000                             # monocular hard trajectory 0 
.
.
|
+-- MH007                             # monocular hard trajectory 7 
```

* [Stereo track](https://drive.google.com/file/d/1dIiN3IxWD_IVVDUKT-BdbX72-lyKUdkh/view?usp=sharing) (Size: 17.51 GB)

  MD5 hash: 8a3363ff2013f147c9495d5bb161c48e

```
stereo
|
--- SE000                                 # stereo easy trajectory 0 
|       |
|       ---- image_left                   # left image folder
|       |       |
|       |       ---- 000000_left.png      # RGB left image 000000
|       |       ---- 000001_left.png      # RGB left image 000001
|       |       .
|       |       .
|       |       ---- 000xxx_left.png      # RGB left image 000xxx
|       |
|       ---- image_right                  # right image folder
|               |
|               ---- 000000_right.png     # RGB right image 000000
|               ---- 000001_right.png     # RGB right image 000001
|               .
|               .
|               ---- 000xxx_right.png     # RGB right image 000xxx
|
+-- SE001                                 # stereo easy trajectory 1 
.
.
+-- SE007                                 # stereo easy trajectory 7 
|
+-- SH000                                 # stereo hard trajectory 0 
.
.
|
+-- SH007                                 # stereo hard trajectory 7 
```

* [Both monocular and stereo tracks](https://drive.google.com/file/d/1N8qoU-oEjRKdaKSrHPWA-xsnRtofR_jJ/view?usp=sharing) (Size: 25.16 GB)

  MD5 hash: ea176ca274135622cbf897c8fa462012 

More information about the [CVPR Visual SLAM challenge](https://sites.google.com/view/vislocslamcvpr2020/slam-challenge)

* The [monocular track](https://www.aicrowd.com/challenges/tartanair-visual-slam-mono-track)

* The [stereo track](https://www.aicrowd.com/challenges/tartanair-visual-slam-stereo-track)

Now the CVPR challenge has completed, the <b> ground truth poses </b> for the above testing trajectories are available [here](https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip). If you need any further support, please send an email to [wenshanw@andrew.cmu.edu](wenshanw@andrew.cmu.edu). 

## Evaluation tools

Following the [TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset) and the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), we adopt three metrics: absolute trajectory error (ATE), the relative pose error (RPE), a modified version of KITTI VO metric. 

[More details](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation)

Check out the sample code: 
```
cd evaluation
python tartanair_evaluator.py
```

Note that our camera poses are defined in the NED frame. That is to say, the x-axis is pointing to the camera's forward, the y-axis is pointing to the camera's right, the z-axis is pointing to the camera's downward. You can use the `cam2ned` function in the `evaluation/trajectory_transform.py` to transform the trajectory from the camera frame to the NED frame. 

## Paper
More technical details are available in the [TartanAir paper](https://arxiv.org/abs/2003.14338). Please cite this as: 
```
@article{tartanair2020iros,
  title =   {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author =  {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year =    {2020}
}
```

## Contact 
Email wenshanw@andrew.cmu.edu if you have any questions about the data source. To post problems in the Github issue is also welcome. You can also reach out to contributors on the associated [GitHub](https://github.com/microsoft/AirSim).

## License
[This software is BSD licensed.](https://opensource.org/licenses/BSD-3-Clause)

Copyright (c) 2020, Carnegie Mellon University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
