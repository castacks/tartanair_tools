DATA_ROOT=/home/chaotec/chessboard/ # /data/datasets/wenshanw/tartan_data
STEP=-1 # 2 #-2
NP=8
DATA_FOLDER='Data,Data_fast'
SAVE_DIR='flow_reverse'
# tartanair_tools/processing/flow_generate/run.sh
# python3 /home/chaotec/tartanair_tools/processing/flow_generate/flow_and_warping_error.py \
python3 process_other_data.py \
    --data-root $DATA_ROOT \
    --data-folders ${DATA_FOLDER} \
    --env-folders '' \
    --index-step $STEP \
    --flow-outdir ${SAVE_DIR} \
    --np $NP\
    # --force-overwrite \
