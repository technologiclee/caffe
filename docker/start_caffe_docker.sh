#!/bin/bash
# A simple script to setup a Caffe build environment. This also ensures that
# the correct user and group are created in the docker container so that caffe
# is not built as root.

### VARIABLES
WHERE_IS_NVIDIA_DOCKER=$(which nvidia-docker)
if [ $? -eq 0 ]
then
  echo "Using nvidia-docker"
  DOCKER=${DOCKER:-"nvidia-docker"}
else
  echo "Using standard docker"
  DOCKER=${DOCKER:-"docker"}
fi

if [ x"$DOCKER" == x"docker" ]
then
  DOCKER_IMAGE="caffe_devel:cpu"
else
  DOCKER_IMAGE="caffe_devel:gpu"
fi


DOCKER_HOST="caffe_devel"
GROUP_ID=$(id -g)
USER_ID=$(id -u)

CAFFE_ROOT=$(pwd)/..

eval ${DOCKER} run \
    -ti \
    -h $DOCKER_HOST \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $CAFFE_ROOT:$CAFFE_ROOT \
    -w $CAFFE_ROOT \
    -e PYTHONPATH=$CAFFE_ROOT/python \
    -u $USER_ID:$GROUP_ID \
    $DOCKER_OPTIONS \
    $DOCKER_IMAGE bash
