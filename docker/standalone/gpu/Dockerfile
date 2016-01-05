FROM nvidia/cuda:7.5-cudnn4-devel

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
    gfortran \
    cmake \
    git

# Use --no-install-recommends for the boost libs.
RUN apt-get install -y --no-install-recommends \
    libboost-all-dev

RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Clone the GIT repository
RUN mkdir -p /opt/caffe && cd /opt/caffe && \
    git clone https://github.com/BVLC/caffe.git . && \
# Install the Python requirements
    cat python/requirements.txt | xargs -n1 pip install && \
# Create a build directory.
    mkdir build && cd build && \    
# Run cmake with the specified arguments.
    cmake -DUSE_CUDNN=1 -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
# Run make
    make -j"$(nproc)" all && \
    make -j"$(nproc)" pycaffe && \
    make -j"$(nproc)" install && \
# Also build utility targets (for testing)
    make -j"$(nproc)" test.testbin


RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
ENV PYTHONPATH=/usr/local/python:$PYTHONPATH

WORKDIR /workspace

# Set the entrypoint to the caffe executable
ENTRYPOINT ["caffe"]
