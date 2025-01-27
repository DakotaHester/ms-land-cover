START=0
END=82

for ((i=START; i<END; i++))
do
    export MSLC_INFERENCE_COUNTY_INDEX=$i
    python inference.py
done