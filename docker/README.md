# Caffe runtime Dockerfiles.

The `runtime` subfolder contains docker files for generating both CPU and GPU runtimes for Caffe. The images can be built using make, or by running:

```
docker build -t caffe runtime/gpu
```
for example.

Note that the GPU runtime requires a CUDA 7.5 capable driver to be installed on the system and [nvidia-docker|https://github.com/NVIDIA/nvidia-docker] for running the Docker containers.

# Running Caffe using the docker image

In order to test the Caffe image, run:
```
docker run -ti caffe --version
```
which should show a message like:
```
libdc1394 error: Failed to initialize libdc1394
caffe version 1.0.0-rc3
```

One can also run the caffe tests using:
```
docker run -ti --entrypoint=/bin/bash caffe -c "cd /opt/caffe/build; make runtest"
```

In order to get the most out of the caffe image, some more advanced `docker run` could be used. For example, running:
```
docker run -ti -v $(pwd):/workspace caffe train --solver=example_solver.prototxt
```
will train a network defined in the `example_solver.prototxt` file in the current directory (`$(pwd)` is maped to the container volume '/workspace' using the `-v` Docker flag).

Note that docker runs all commands as root by default, and thus any output files (e.g. snapshots) generated will be owned by the root user. In order to ensure that the current user is used instead, the following command can be used:
```
docker run -ti -v $(pwd):/workspace -u $(id -u):$(id -g) caffe train --solver=example_solver.prototxt
```
where the `-u` Docker command line option runs the commands in the container as the specified user, and the shell command `id` is used to determine the user and group ID of the current user.


# Caffe development Dockerfile.

The files contained here allow for the Docker images to be built which contain
the development environment for Caffe.

In order to use GPU computing with docker, nvidia-docker (https://github.com/NVIDIA/nvidia-docker) is recommended. The Docker image uses the NVIDIA CUDA
image as a starting point to allow for GPU accelleration within Docker.

# Usage

First ensure that the docker image is built:

```make docker_devel```

This will create a docker image with the tag ```caffe:devel``` which can be
used with Docker as per usual.

A utility script is also provided to start a container based on this image.
This container can be used to build and run caffe.

To use the script run:

```./start_caffe_docker.sh bash```

This should show the following output:

```
Using nvidia-docker
[ NVIDIA ] =INFO= Driver version: 352.63
[ NVIDIA ] =INFO= CUDA image version: 7.5

elezar@caffe_devel:~/caffe$
```
Where this the NVIDIA docker wrapper is used in this case, and ```elezar``` is
the username of the user running docker.

The caffe source folder is mounted as a volume in the container at ```~/caffe```.