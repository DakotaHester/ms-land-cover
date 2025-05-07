#!/bin/bash
# START=0
# # END=82
# END=755
# OUT_FOLDER='./data/inference_results_v2/MS/*'
# TARGET_FOLDER='/home/dhester/server/guser/dh/MS_HiRes_LC_Prelim_v2/'

# for ((i=START; i<END; i++))
# do
#     export MSLC_INFERENCE_COUNTY_INDEX=$i
#     python inference.py
# done


START=0
# START=118
END=755
OUT_FOLDER='./data/inference_results_v2/MS/*'
TARGET_FOLDER='/home/dhester/server/guser/dh/MS_HiRes_LC_Prelim_v2/'

MAX_JOBS=4  # Number of concurrent processes

for ((i=START; i<END; i++))
do
    export MSLC_INFERENCE_COUNTY_INDEX=$i
    # python inference.py &  # Run in background
    # pids+=($!)             # Track background process IDs
    python inference.py

    # # Wait when MAX_JOBS background processes are running
    # if (( ${#pids[@]} >= MAX_JOBS )); then
    #     wait -n           # Wait for any one job to finish
    #     # Remove finished job PIDs from the array
    #     new_pids=()
    #     for pid in "${pids[@]}"; do
    #         if kill -0 "$pid" 2>/dev/null; then
    #             new_pids+=("$pid")
    #         fi
    #     done
    #     pids=("${new_pids[@]}")
    # fi
done

# Wait for remaining background jobs to complete
wait
