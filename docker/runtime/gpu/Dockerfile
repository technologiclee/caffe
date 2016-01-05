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
# Clone Caffe repo and move into it
RUN cd /opt && git clone https://github.com/BVLC/caffe.git && cd caffe && \
# Get the python prerequsites for this repository
  cat python/requirements.txt | xargs -n1 pip install && \
# Create a build directory
  mkdir build && cd build && \
# Run cmake
  cmake -DUSE_CUDNN=1 .. && \
# # Make
  make -j"$(nproc)" all && \
  make -j"$(nproc)" pycaffe
# Also build some other utility targets.
RUN cd /opt/caffe/build && make -j"$(nproc)" test.testbin


# Add to Python path
ENV CAFFE_ROOT=/opt/caffe

ENV PYTHONPATH=$CAFFE_ROOT/python:$PYTHONPATH
ENV PATH=$CAFFE_ROOT/build/tools:$PATH

# Set /workspace as working directory
WORKDIR /workspace

# Set the entrypoint
ENTRYPOINT ["caffe"]

# And the default command line options.
CMD ["--help"]
