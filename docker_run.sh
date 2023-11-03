#!/bin/bash

# Define the image name and tag
IMAGE_NAME="assignment"
TAG="4"

# Build the Docker image
docker build -t "$IMAGE_NAME:$TAG" -f docker/Dockerfile .

# Run the Docker container with the VOLUME
docker run -v $(pwd)/models:/models "$IMAGE_NAME:$TAG"
