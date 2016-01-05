FROM nvidia/cuda:7.5-cudnn4-devel
MAINTAINER team-sl@zalando.de

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgflags-dev \
    libgoogle-glog-dev \
    libopencv-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libhdf5-serial-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libatlas-base-dev \
    python-dev \
    python-pip \
    python-numpy \
    python-scipy \
    gfortran

# Use --no-install-recommends for the boost libs.
RUN apt-get install -y --no-install-recommends \
    libboost-all-dev

# Install utilities used for the build.
RUN apt-get install -y \
    build-essential \
    cmake \
    cmake-curses-gui \
    git \
    ca-certificates \
    bc \
    wget

# Install the python requirements.
RUN wget https://raw.githubusercontent.com/BVLC/caffe/master/python/requirements.txt -O /tmp/requirements.txt && \
    cat /tmp/requirements.txt && \
    cat /tmp/requirements.txt | xargs -n1 pip install

# Clean up the apt-get cache and remove unused packages.
RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*

CMD bash
