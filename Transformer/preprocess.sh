#!/usr/bin/env bash

DATA_PATH=data
DATASET=$1
METADATA=$2

python data.py --dataset $DATASET --metadata $METADATA

python preprocess.py \
--text-path $DATA_PATH/$DATASET/train_texts.txt \
--label-path $DATA_PATH/$DATASET/train_labels.txt \
--vocab-path $DATA_PATH/$DATASET/vocab.npy \
--emb-path $DATA_PATH/$DATASET/emb_init.npy \
--w2v-model $DATA_PATH/glove.6B.100d.txt

python preprocess.py \
--text-path $DATA_PATH/$DATASET/test_texts.txt \
--label-path $DATA_PATH/$DATASET/test_labels.txt \
--vocab-path $DATA_PATH/$DATASET/vocab.npy
