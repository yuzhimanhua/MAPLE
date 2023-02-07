#!/bin/bash

dataset=Art
python match.py --dataset $dataset
# dataset=Biology_MeSH
# python match_mesh.py --dataset $dataset

metadata=None
python encode.py --dataset $dataset --metadata $metadata
cd Parabel/
./sample_run.sh $dataset
cd ../

metadata=Venue
python encode.py --dataset $dataset --metadata $metadata
cd Parabel/
./sample_run.sh $dataset
cd ../

metadata=Author
python encode.py --dataset $dataset --metadata $metadata
cd Parabel/
./sample_run.sh $dataset
cd ../