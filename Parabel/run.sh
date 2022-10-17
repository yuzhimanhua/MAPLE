#!/bin/bash

dataset=Art

metadata=None
python data.py --dataset $dataset --metadata $metadata
./sample_run.sh $dataset

metadata=Venue
python data.py --dataset $dataset --metadata $metadata
./sample_run.sh $dataset

metadata=Author
python data.py --dataset $dataset --metadata $metadata
./sample_run.sh $dataset

metadata=Reference
python data.py --dataset $dataset --metadata $metadata
./sample_run.sh $dataset
