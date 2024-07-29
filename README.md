# Docker Compose Setup for Model Training and Inference

This is the easiest way to distribute and train/infer a model, as the user only needs to have Docker installed and run `docker compose up [inference, train]`.

## Overview

This setup consists of two services defined in the `docker-compose.yml` file:

1. **train**: A service to train the model.
2. **inference**: A service to run inference on sample photos in `./inference/input` and write the masks to `./inference/output`.

Both services use the same Dockerfile and leverage multi-stage building for efficient caching. The base image is built from source to accommodate compatibility with the Mac Air M2 architecture, as the official TensorFlow image is only built for amd64 and would throw errors related to incorrect CPU instructions when run on this architecture.

## Project Layout

The project is organized into several directories:

- **data**: Contains all the dataset files. This folder is untracked by git due to the large size of the data.
- **inference**: Contains the Dockerfile and `inference.py` script for running inference.
- **jupyter**: Contains Jupyter notebooks for analysis and other exploratory tasks.
- **model**: Contains the trained model files. This folder is tracked by git as the model files are small.
- **train**: Contains the Dockerfile and `train.py` script for training the model.
- **venv**: Contains virtual environment-related files and configurations.

Additionally, the project includes various build and configuration files such as `.python-version`, `requirements.txt`, `.dockerignore`, and `docker-compose.yml`.

## Docker Compose File

```yaml
version: '3.8'

services:
  train:
    build:
      context: .
      dockerfile: train/Dockerfile
    container_name: train_service
    volumes:
      - ./data:/app/data
      - ./model:/app/model

  inference:
    build:
      context: .
      dockerfile: inference/Dockerfile
    container_name: inference_service
    volumes:
      - ./data:/app/data
      - ./model:/app/model
      - ./inference/input:/app/input
      - ./inference/output:/app/output
    environment:
      - INPUT_FOLDER=/app/input/
      - OUTPUT_FOLDER=/app/output/
      - MODEL_PATH=/app/model/unet_model.keras
```

## Dockerfile

The Dockerfile used for both services employs multi-stage builds to separate the dependencies installation and the final image preparation. This method ensures that the images are built efficiently with proper caching.

```dockerfile
# Stage 1: Base image
FROM python:3.12-bookworm AS base

# Install necessary packages
RUN apt-get update && apt-get install -y build-essential \
                        curl \
                        git \
                        unzip \
                        zip \
                        libfreetype6-dev \
                        libhdf5-dev \
                        libzmq3-dev \
                        pkg-config \
                        software-properties-common \
                        libffi-dev \
                        libssl-dev \
                        zlib1g-dev \
                        liblcms2-dev \
                        libblas-dev \
                        liblapack-dev \
                        gfortran \
                        libpng-dev \
                        libreadline-dev \
                        libsqlite3-dev \
                        libncurses5-dev \
                        libncursesw5-dev \
                        xz-utils \
                        tk-dev \
                        libgdbm-dev \
                        libc6-dev \
                        libbz2-dev \
                        cmake \
                        libjpeg-dev \
                        libtiff-dev \
                        libavcodec-dev \
                        libavformat-dev \
                        libswscale-dev \
                        libv4l-dev \
                        libxvidcore-dev \
                        libx264-dev \
                        libgtk-3-dev \
                        libatlas-base-dev \
                        python3-dev

# Install Python dependencies
RUN apt-get install -y python3-dev python3-pip python3-venv
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel && python3 -m pip install --no-cache-dir wheel six numpy packaging h5py

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build the final training image
FROM base AS train

WORKDIR /app

COPY inference/inference.py /app/

CMD ["python", "inference.py"]
```

## Key Points

- **Multi-Stage Building**: This approach allows for more efficient caching and smaller final images.
- **Custom Base Image**: The base image is built almost from source to include all necessary package dependencies and Python libraries required for training and inference.
- **Architecture Compatibility**: The custom base image ensures compatibility with Mac Air M2 architecture, avoiding issues related to incorrect CPU instructions when using the official TensorFlow image built for amd64.

By using this setup, you can easily manage and run both training and inference services with Docker, ensuring compatibility and efficient resource usage.
