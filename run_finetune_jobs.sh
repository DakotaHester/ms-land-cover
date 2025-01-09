#!/bin/bash

# Constants
MAX_CONCURRENT_JOBS=8   # Maximum number of concurrent jobs
MAX_TASK_ID=154         # Total number of tasks (0 to 153)
LOG_DIR="./logs/finetune/out" # Log directory
mkdir -p "$LOG_DIR"
source ~/ms-land-cover/.env/bin/activate

# Function to count running Python processes for this script
count_running_jobs() {
    ps -u "$USER" -o comm,args | grep -E "python finetune\.py --num_workers" | grep -v grep | wc -l
}

# Task submission loop
for TASK_ID in $(seq 0 $MAX_TASK_ID); do
    while true; do
        # Count running jobs
        RUNNING_JOBS=$(count_running_jobs)

        if (( RUNNING_JOBS < MAX_CONCURRENT_JOBS )); then
            echo "Starting task $TASK_ID (Running Jobs: $RUNNING_JOBS)"

            # Set TASK_ARRAY_ID and start the Python script
            export TASK_ARRAY_ID=$TASK_ID
            nohup python finetune.py --num_workers 4 > "$LOG_DIR/task_${TASK_ID}.log" 2>&1 &

            # Log the process start
            echo "Task $TASK_ID started successfully." >> "$LOG_DIR/tasks_started.log"
            
            # Break the loop to proceed with the next task
            break
        else
            echo "Maximum concurrent jobs reached ($RUNNING_JOBS/$MAX_CONCURRENT_JOBS). Waiting..."
            sleep 5
        fi
    done
done

# Wait for all jobs to finish
while (( $(count_running_jobs) > 0 )); do
    echo "Waiting for remaining jobs to finish..."
    sleep 5
done

# Log completion
echo "All tasks have been scheduled and completed."
