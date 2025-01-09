#!/bin/bash
for ((TASK_ARRAY_ID=0; TASK_ARRAY_ID<154; TASK_ARRAY_ID++));
do
    export TASK_ARRAY_ID
    python finetune.py
done