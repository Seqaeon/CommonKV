#!/bin/bash

export job_name=granite-ctrl-llm-1g

bsub \
        -J $job_name \
        -gpu \"num=1/task:mode=exclusive_process\" \
        -n 1 \
        -M 512G \
        -W 120:00 \
        -o ./bv_output/${job_name}-%J-$1.stdout \
        -e ./bv_output/${job_name}-%J-$1.stderr \
        < $1

exit 0
        #-q standard \
