#!/usr/bin/env sh
# The following example allows for the MNIST example (using LeNet) to be run
# using the caffe docker image instead of building from source.
# The GPU-enabled version of Caffe can be used, assuming that nvidia-docker
# is installed, and the GPU-enabled Caffe image has been built.
# Setting the GPU environment variable to 1 will enable the use of nvidia-docker.
# e.g.   
#   GPU=1 ./examples/mnist/train_lenet_docker.sh
# 
# Not the use of the -u, -v, and -w command line options to ensure that 
# files are created under the ownership of the current user, and in the 
# current directory.

if [ $GPU -ne 1 ]
then
DOCKER_CMD=docker
else
DOCKER_CMD=nvidia-docker
fi

$DOCKER_CMD run --rm -ti \
    -u $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    -w /workspace \
    caffe:runtime train --solver=examples/mnist/lenet_solver.prototxt $*
