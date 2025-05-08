#!/bin/bash

# Define variables
DOCKER_TAG=aicregistry:5000/${USER}:sworld


# Build the Docker image
docker build . -f Dockerfile \
 --network=host \
 --tag ${DOCKER_TAG} \
 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER}

# Push the Docker image to Docker Hub
docker push $DOCKER_TAG