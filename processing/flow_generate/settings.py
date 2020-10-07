import argparse

def get_args():
    parser = argparse.ArgumentParser(description='sample_pipeline')

    # pipeline control
    parser.add_argument('--mapping', action='store_true', default=False,
                        help='mapping the environment (default: False)')

    parser.add_argument('--sample-graph', action='store_true', default=False,
                        help='sample graph (default: False)')

    parser.add_argument('--sample-path', action='store_true', default=False,
                        help='sample path (default: False)')

    parser.add_argument('--sample-position', action='store_true', default=False,
                        help='sample position (default: False)')

    parser.add_argument('--data-collection', action='store_true', default=False,
                        help='data collection (default: False)')

    # mapping - expo_control
    parser.add_argument('--map-dir', default='~/tmp',
                        help='output map file directory')

    parser.add_argument('--map-filename', default='map',
                        help='output map file name')

    parser.add_argument('--path-skip', type=int, default=7,
                        help='skip steps on the path, the bigger the faster (default: 7)')

    parser.add_argument('--global-only', action='store_true', default=False,
                        help='only visit global frontiers, fast but inaccurate (default: False)')

    parser.add_argument('--camera-fov', type=float, default=90.0,
                        help='camera fov (default: 100)')

    parser.add_argument('--far-point', type=int, default=22,
                        help='a number a little larger than the Lidar range (default: 22)')

    parser.add_argument('--try-round', type=int, default=-1,
                        help='A* planning round (default: unlimited)')


    # Graph sampling
    parser.add_argument('--environment-dir', default='/home/wenshan/tmp/maps/',
                        help='graph output dir ')

    parser.add_argument('--graph-filename', default='',
                        help='graph file name ')

    parser.add_argument('--prune-node-num', type=int, default=0,
                        help='prune the graph with min node number after loaded (default: 0)')

    # Graph sampling - step 1 sample graph
    parser.add_argument('--node-num', type=int, default=30,
                        help='number of nodes to be sampled (default: 100)')

    parser.add_argument('--edge-num', type=int, default=8,
                        help='number of edges for each node (default: 10)')

    parser.add_argument('--min-dist-thresh', type=int, default=5,
                        help='minimum distance of two linked nodes (default: 10)')

    parser.add_argument('--max-dist-thresh', type=int, default=20,
                        help='maximum distance of two linked nodes (default: 30)')

    parser.add_argument('--max-failure-num', type=int, default=20,
                        help='maximum number of planning failure before giveup (default: 20)')

    parser.add_argument('--node-range-xmin', type=float, default=-10.0,
                        help='node sample range (default: -10.0)')

    parser.add_argument('--node-range-xmax', type=float, default=10.0,
                        help='node sample range (default: 10.0)')

    parser.add_argument('--node-range-ymin', type=float, default=-10.0,
                        help='node sample range (default: -10.0)')

    parser.add_argument('--node-range-ymax', type=float, default=10.0,
                        help='node sample range (default: 10.0)')

    parser.add_argument('--node-range-zmin', type=float, default=-2.0,
                        help='node sample range (default: -2.0)')

    parser.add_argument('--node-range-zmax', type=float, default=2.0,
                        help='node sample range (default: 2.0)')

    # Graph sampling - step 2 sample cycle

    parser.add_argument('--ros-path-surfix', default='',
                        help='surfix of the output folder ')

    parser.add_argument('--sample-cycle-mode', type=int, default=0,
                        help='0 random, 1 min, 2 max (default: 0)')


    # Graph sampling - step 3 sample position
    parser.add_argument('--position-path-surfix', default='',
                        help='surfix of the position folder ')

    parser.add_argument('--ros-path-dir', default='',
                        help='for sample position on path output dir ')

    parser.add_argument('--dist-max', type=float, default=0.3,
                        help='max distance (default: 100)')

    parser.add_argument('--dist-min', type=float, default=0.0,
                        help='mim distance (default: 100)')

    parser.add_argument('--acc-max', type=float, default=0.1,
                        help='max acceleration (default: 100)')

    parser.add_argument('--step-max', type=int, default=20,
                        help='max stepsize before change acceleration (default: 100)')

    # image collection - collect_images
    parser.add_argument('--position-folder', default='',
                        help='folder for position files (default: )')

    parser.add_argument('--data-folder-perfix', default='',
                        help='folder perfix (default: )')

    parser.add_argument('--cam-list', default='1_2',
                        help='camera list: 0-front, 1-right, 2-left, 3-back, 4-bottom (default: 1_2)')

    parser.add_argument('--img-type', default='Scene_DepthPlanner_Segmentation',
                        help='image type Scene, DepthPlanner, Segmentation (default: Scene_DepthPlanner_Segmentation)')

    parser.add_argument('--rand-degree', type=int, default=30,
                        help='random angle added to the position when sampling (default: 15)')

    parser.add_argument('--smooth-count', type=int, default=10,
                        help='lengh of smoothed trajectory (default: 10)')

    parser.add_argument('--max-yaw', type=int, default=360,
                        help='yaw threshold (default: 360)')

    parser.add_argument('--min-yaw', type=int, default=-360,
                        help='yaw threshold (default: -360)')

    parser.add_argument('--max-pitch', type=int, default=20,
                        help='yaw threshold (default: 45)')

    parser.add_argument('--min-pitch', type=int, default=-45,
                        help='yaw threshold (default: -45)')

    parser.add_argument('--max-roll', type=int, default=20,
                        help='yaw threshold (default: 90)')

    parser.add_argument('--min-roll', type=int, default=-20,
                        help='yaw threshold (default: -90)')

    parser.add_argument('--load-posefile', action='store_true', default=False,
                        help='load both position and orientation from file')

    parser.add_argument('--load-posefile-left-right', action='store_true', default=False,
                        help='load both position and orientation from left and right pose files')

    parser.add_argument('--gamma', type=float, default=3.7,
                        help='gamma in airsim settings.json')

    parser.add_argument('--min-exposure', type=float, default=0.3,
                        help='MinExposure in airsim settings.json')

    parser.add_argument('--max-exposure', type=float, default=0.7,
                        help='MaxExposure in airsim settings.json')

    parser.add_argument('--save-posefile-only', action='store_true', default=False,
                        help='only sample the poses without save the image data')

    # image validation
    parser.add_argument('--data-root', default='/data/datasets/wenshanw/tartan_data',
                        help='root data folder that contrains environment folders')

    parser.add_argument('--env-folders', default='',
                        help='specify the environment folder, all the folders if not specified')

    parser.add_argument('--data-folders', default='Data,Data_fast',
                        help='data folders in each environment folder')

    parser.add_argument('--create-video', action='store_true', default=False,
                        help='generate preview video (default: False)')

    parser.add_argument('--video-with-flow', action='store_true', default=False,
                        help='save flow in the video instead of right image (default: False)')

    parser.add_argument('--analyze-depth', action='store_true', default=False,
                        help='calculate depth statistics from depth image and output files (default: False)')

    parser.add_argument('--depth-from-file', action='store_true', default=False,
                        help='read depth info from files, only active when --analyze-depth is set (default: False)')

    parser.add_argument('--rgb-validate', action='store_true', default=False,
                        help='read rgb image and output statistics file (default: False)')

    parser.add_argument('--depth-filter', action='store_true', default=False,
                        help='filter depth and generate text file for stereo training (default: False)')

    parser.add_argument('--rgb-depth-filter', action='store_true', default=False,
                        help='filter depth and rgb value and generate text file for stereo training (default: False)')

    # optical flow and warping error
    parser.add_argument("--np", type=int, default=1, 
                        help="Number of processes.")

    # --data-root, --env-folders, --data-folders are shared with data valiation

    parser.add_argument("--index-step", type=int, default=1, 
                        help="Generate optical flow for every STEP ")

    parser.add_argument("--start-index", type=int, default=0, 
                        help="Skip the first few images ")

    parser.add_argument("--focal", type=int, default=320, 
                        help="camera focal length")

    parser.add_argument("--image-width", type=int, default=640, 
                        help="image width")

    parser.add_argument("--image-height", type=int, default=480, 
                        help="image height")

    parser.add_argument('--save-flow-image', action='store_true', default=False,
                        help='save optical flow image for debugging')

    parser.add_argument('--force-overwrite', action='store_true', default=False,
                        help='save optical flow in a same flow folder')

    parser.add_argument('--flow-outdir', default='flow',
                        help='output flow file to this folder (default: flow)')

    parser.add_argument('--target-root', default='',
                        help='copying flow to another drive, target root data folder that contrains environment folders')

    args = parser.parse_args()

    return args
