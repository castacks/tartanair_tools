DATA_ROOT=/data/datasets/wenshanw/tartan_data
DATA_FOLDER='Data,Data_fast'
LEFT_RIGHT='image_left'
# LEFT_RIGHT='image_right'
# python3 motion_blur.py \
NUM_WORKERS=16
python3 /home/chaotec/tartanair_tools/processing/motion_blur.py \
    --workers $NUM_WORKERS\
    --data_root $DATA_ROOT \
    --data_types ${DATA_FOLDER} \
    --left_right ${LEFT_RIGHT}\
