# Stage 1: Base image
FROM python:3.12-bookworm AS base
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



RUN apt-get install -y python3-dev python3-pip python3-venv
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel && python3 -m pip install --no-cache-dir wheel six numpy packaging h5py

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Build the final training image
FROM base AS train

WORKDIR /app
RUN mkdir input output

COPY inference/inference.py /app/

CMD ["python", "inference.py"]
