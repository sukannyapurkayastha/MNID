export PERIOD=1
export DATASET=hwu_ood
export LOGFILE=../DataProcessed/$DATASET/results_BERT_P$PERIOD.txt
export GPU=0

nohup python3 -u main.py \
    --gpu $GPU \
    --dataset $DATASET \
    --period $PERIOD > $LOGFILE &

tail -f $LOGFILE
