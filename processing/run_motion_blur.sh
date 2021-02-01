DATA_ROOT=/data/datasets/wenshanw/tartan_data
DATA_FOLDER='Data,Data_fast'
LEFT_RIGHT='image_left'
EXPOSURE_RATE=0.5 # 1.0
NUM_WORKERS=16 # 2
NUM_SPLIT=3
SPLIT_ID=2

# LEFT_RIGHT='image_right'
# python3 /home/chaotec/tartanair_tools/processing/motion_blur.py \
# python3 motion_blur.py \
python3 /home/chaotec/tartanair_tools/processing/motion_blur.py \
    --workers $NUM_WORKERS\
    --data_root $DATA_ROOT \
    --data_types ${DATA_FOLDER} \
    --left_right ${LEFT_RIGHT}\
    --exposure_rate $EXPOSURE_RATE\
    --num_split $NUM_SPLIT\
    --split_id $SPLIT_ID
## run multiple different EXPOSURE_RATE
# for EXPOSURE_RATE in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
# do
#     python3 /home/chaotec/tartanair_tools/processing/motion_blur.py --workers $NUM_WORKERS --data_root $DATA_ROOT --data_types ${DATA_FOLDER} --left_right ${LEFT_RIGHT} --exposure_rate $EXPOSURE_RATE --num_split $NUM_SPLIT --split_id $SPLIT_ID
    # python3 motion_blur.py --workers $NUM_WORKERS --data_root $DATA_ROOT --data_types ${DATA_FOLDER} --left_right ${LEFT_RIGHT} --exposure_rate $EXPOSURE_RATE
# done
