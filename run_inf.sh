#!/bin/bash
START=0
END=82
OUT_FOLDER='./data/inference_results/MS/*'
TARGET_FOLDER='/home/dhester/server/guser/dh/MS_HiRes_LC_Prelim/'

for ((i=START; i<END; i++))
do
    export MSLC_INFERENCE_COUNTY_INDEX=$i
    python inference.py
    mv $OUT_FOLDER $TARGET_FOLDER
done