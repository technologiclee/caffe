# A makefile to build the docker images for caffe.

all: docker_files devel runtime

.PHONY: runtime devel

runtime: devel cpu_runtime gpu_runtime
devel: cpu_devel gpu_devel


cpu_runtime: runtime/cpu/Dockerfile
	docker build -t caffe runtime/cpu

gpu_runtime: runtime/gpu/Dockerfile
	docker build -t caffe:gpu runtime/gpu

gpu_devel: devel/gpu/Dockerfile
	docker build -t caffe_devel:gpu devel/gpu

cpu_devel: devel/cpu/Dockerfile
	docker build -t caffe_devel:cpu devel/cpu

docker_files: devel_files runtime_files

devel_files: devel/cpu/Dockerfile devel/gpu/Dockerfile
runtime_files: runtime/cpu/Dockerfile runtime/gpu/Dockerfile


from_gpu = "FROM nvidia/cuda:7.5-cudnn4-devel"
from_cpu = "FROM ubuntu:14.04"

define create_docker_file
	@echo creating $@
	@echo $(if $(strip $(findstring gpu,$@)),$(from_gpu),$(from_cpu)) > $@
	@echo "MAINTAINER team-sl@zalando.de" >> $@
	@echo "" >> $@
	@cat $^ >> $@
endef


devel/%/Dockerfile: templates/devel.template
	$(create_docker_file)

runtime/%/Dockerfile: templates/devel.template templates/%_build.template templates/runtime.template
	$(create_docker_file)

