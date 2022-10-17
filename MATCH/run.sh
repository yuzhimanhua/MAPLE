#!/usr/bin/env bash

DATASET=Art
LABEL=All

METADATA=None
./preprocess.sh $DATASET $METADATA $LABEL
./run_models.sh $DATASET
# rm -r data/$DATASET/

METADATA=Venue
./preprocess.sh $DATASET $METADATA $LABEL
./run_models.sh $DATASET
# rm -r data/$DATASET/

METADATA=Author
./preprocess.sh $DATASET $METADATA $LABEL
./run_models.sh $DATASET
# rm -r data/$DATASET/

METADATA=Reference
./preprocess.sh $DATASET $METADATA $LABEL
./run_models.sh $DATASET
# rm -r data/$DATASET/
