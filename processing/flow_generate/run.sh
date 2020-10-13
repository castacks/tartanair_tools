DATA_ROOT=/data/datasets/wenshanw/tartan_data
STEP=-1 # 2 #-2
NP=1 # 8
DATA_FOLDER='Data,Data_fast'
python3 flow_and_warping_error.py \
    --data-root $DATA_ROOT \
    --data-folders ${DATA_FOLDER} \
    --env-folders '' \
    --index-step $STEP \
    --flow-outdir flow2 \
    --np $NP
