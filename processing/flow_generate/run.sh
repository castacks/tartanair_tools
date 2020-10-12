DATA_ROOT=/data/datasets/wenshanw/tartan_data
STEP=2
DATA_FOLDER=Data
python3 flow_and_warping_error.py \
    --data-root $DATA_ROOT \
    --data-folders $DATA_FOLDER \
    --env-folders '' \
    --index-step $STEP \
    --flow-outdir flow2 \
    --np 8
