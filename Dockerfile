FROM ubuntu:14.04
MAINTAINER Felix Lau <felixlaumon@gmail.com>

# From https://registry.hub.docker.com/u/tleyden5iwx/ubuntu-cuda/dockerfile/

# Download required files
RUN apt-get update && apt-get install -q -y wget

ENV CUDA_RUN http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
ENV CUDNN_TAR https://www.dropbox.com/s/1zstyz9n5t7lrtt/cudnn-6.5-linux-R1.tgz?dl=1

RUN cd /opt && wget $CUDA_RUN && chmod +x *.run
# RUN wget http://repo.continuum.io/miniconda/Miniconda-3.7.0-Linux-x86_64.sh -O miniconda.sh
RUN wget --quiet http://repo.continuum.io/archive/Anaconda-2.1.0-Linux-x86_64.sh -O anaconda.sh
RUN cd /opt && \
    wget $CUDNN_TAR -O cudnn-6.5-linux-R1.tgz && \
    tar xvfz cudnn-6.5-linux-R1.tgz

# Update and install dependencies
RUN apt-get install -q -y \
    build-essential \
    vim \
    locales \
    curl \
    unzip \
    openssl \
    locate \
    git

# Install Nvidia driver and CUDA
RUN cd /opt && \
    mkdir nvidia_installers && \
    ./cuda_6.5.14_linux_64.run -extract=`pwd`/nvidia_installers && \
    cd nvidia_installers && \
    ./NVIDIA-Linux-x86_64-340.29.run -s -N --no-kernel-module
RUN cd /opt/nvidia_installers && \
  ./cuda-linux64-rel-6.5.14-18749181.run -noprompt
ENV PATH /usr/local/cuda-6.5/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-6.5/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda-6.5

# CUDNN
ENV LD_LIBRARY_PATH /cudnn-6.5-linux-R1/:$LD_LIBRARY_PATH
ENV CPATH /cudnn-6.5-linux-R1/:$CPATH
ENV LIBRARY_PATH /cudnn-6.5-linux-R1/:$LIBRARY_PATH

# Install miniconda and some python dependencies
# RUN bash miniconda.sh -b -p /miniconda && rm miniconda.sh
# ENV PATH /miniconda/bin:$PATH
RUN /bin/bash /anaconda.sh -b -p /opt/conda && \
    rm /anaconda.sh && \
    /opt/conda/bin/conda install --yes conda==3.9.0
ENV PATH /opt/conda/bin:$PATH
RUN conda install --yes \
    scipy \
    numpy \
    scikit-learn \
    scikit-image \
    pyzmq \
    nose \
    readline \
    pandas \
    matplotlib \
    seaborn \
    dateutil \
    ipython-notebook \
    pip

# Actual code
ADD . /plankton
WORKDIR /plankton
RUN pip install -r requirements.txt
RUN pip install -r requirements2.txt

# Tell theano to use GPU
ENV THEANO_FLAGS floatX=float32,device=gpu0,nvcc.fastmath=True
