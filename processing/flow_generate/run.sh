DATA_ROOT=/data/datasets/wenshanw/tartan_data
STEP=-1 # 2 #-2
NP=16
DATA_FOLDER='Data,Data_fast'
SAVE_DIR='flow_reverse_right'
EYE='right'
NUM_SPLIT=3
SPLIT_ID=0
# tartanair_tools/processing/flow_generate/run.sh
# python3 /home/chaotec/tartanair_tools/processing/flow_generate/flow_and_warping_error.py \
python3 flow_and_warping_error.py \
    --data-root $DATA_ROOT \
    --data-folders ${DATA_FOLDER} \
    --env-folders '' \
    --index-step $STEP \
    --flow-outdir ${SAVE_DIR} \
    --np $NP\
    --num_split $NUM_SPLIT\
    --split_id $SPLIT_ID\
    --eye ${EYE}\
    --force-overwrite \
