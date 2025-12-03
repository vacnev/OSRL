# Use NVIDIA CUDA base image with cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MUJOCO_PY_MUJOCO_PATH=/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
ENV MUJOCO_GL=osmesa

# Install Python 3.8 and mujoco deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3.10-dev \
    python3-dev \
    build-essential \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libxrender1 libxext6 libsm6 \
    patchelf \
    gcc \
    cmake \
    wget \
    git \
    swig \
    zlib1g-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libavutil56 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo
RUN mkdir -p /.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar --no-same-owner -xzf mujoco.tar.gz -C /.mujoco \
    && rm mujoco.tar.gz

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Pre-install numpy to resolve OApackage build dependency issue
RUN pip install numpy
RUN pip install --no-build-isolation OApackage==2.7.6