# A makefile to build the docker images for caffe.

DOCKER ?= docker

all: docker_files standalone

.PHONY: standalone devel

standalone: cpu_standalone gpu_standalone


cpu_standalone: standalone/cpu/Dockerfile
	$(DOCKER) build -t caffe standalone/cpu

gpu_standalone: standalone/gpu/Dockerfile
	$(DOCKER) build -t caffe:gpu standalone/gpu

docker_files: standalone_files

standalone_files: standalone/cpu/Dockerfile standalone/gpu/Dockerfile

from_gpu = "nvidia/cuda:7.5-cudnn4-devel"
from_cpu = "ubuntu:14.04"
gpu_cmake_args = -DUSE_CUDNN=1
cpu_cmake_args = -DCPU_ONLY=1

# A make macro to select the CPU or GPU base image.
define from_image
$(if $(strip $(findstring gpu,$@)),$(from_gpu),$(from_cpu))
endef

# A make macro to select the CPU or GPU build args.
define build_args
$(if $(strip $(findstring gpu,$@)),$(gpu_cmake_args),$(cpu_cmake_args))
endef

# A make macro to construct the CPU or GPU Dockerfile from the template
define create_docker_file
	@echo creating $@
	@echo "FROM "$(from_image) > $@	
	@cat $^ | sed 's/$${CMAKE_ARGS}/$(build_args)/' >> $@
endef


standalone/%/Dockerfile: templates/Dockerfile.template
	$(create_docker_file)

