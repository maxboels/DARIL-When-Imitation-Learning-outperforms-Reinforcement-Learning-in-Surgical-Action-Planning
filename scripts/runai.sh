#!/bin/bash

# Define variables
CONFIG_NAME=$1
JOB_NAME=$(echo $CONFIG_NAME | cut -f 1 -d '.' | tr '_' '-' | tr '/' '-')
DOCKER_IMAGE="aicregistry:5000/mboels:swrl"
PROJECT_NAME="mboels"
NUM_GPUS=1
NUM_CPUS=2
VOLUME_MOUNT="/nfs:/nfs"
COMMAND="bash /nfs/home/$USER/projects/surl/run_file.sh $CONFIG_NAME"

# If job name exists, delete it
runai list | grep $JOB_NAME && runai delete job $JOB_NAME
sleep 3

# Submit the Run:AI job
runai submit $JOB_NAME \
       --image $DOCKER_IMAGE \
       --run-as-user \
       --large-shm \
       --gpu $NUM_GPUS \
       --cpu $NUM_CPUS \
       --project $PROJECT_NAME \
       -v $VOLUME_MOUNT \
       --backoff-limit 0 \
       --command -- $COMMAND

# Wait for 3 seconds and list the jobs
sleep 3
watch runai list