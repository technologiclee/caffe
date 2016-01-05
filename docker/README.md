# Caffe standalone Dockerfiles.

The `standalone` subfolder contains docker files for generating both CPU and GPU executable images for Caffe. The images can be built using make, or by running:

```
docker build -t caffe standalone/gpu
```
for example.

Note that the GPU standalone requires a CUDA 7.5 capable driver to be installed on the system and [nvidia-docker|https://github.com/NVIDIA/nvidia-docker] for running the Docker containers.

# Running Caffe using the docker image

In order to test the Caffe image, run:
```
docker run -ti caffe --version
```
or 
```
nvidia-docker run -ti caffe:gpu --version
```
which should both show a message like:
```
libdc1394 error: Failed to initialize libdc1394
caffe version 1.0.0-rc3
```

One can also run the caffe tests using:
```
docker run -ti --entrypoint=/bin/bash caffe -c "cd /opt/caffe/build; make runtest"
```
or
```
nvidia-docker run -ti --entrypoint=/bin/bash caffe:gpu -c "cd /opt/caffe/build; make runtest"
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
