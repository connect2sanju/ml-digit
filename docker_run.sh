#!/bin/bash

# Define the image name and tag
IMAGE_NAME="assignment"
TAG="4"
# container_name="sanjib_container"

# Build the Docker image
docker build -t "$IMAGE_NAME:$TAG" -f docker/Dockerfile .

# echo "Host's models directory: $(pwd)/models"

# Run the Docker container with the VOLUME
docker run -v $(pwd)/models:/digits/models --name container2 "$IMAGE_NAME:$TAG"

# Stop and remove the container
# docker stop "$container_name"
# docker rm "$container_name"