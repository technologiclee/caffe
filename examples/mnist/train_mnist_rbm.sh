#!/usr/bin/env sh

./build/tools/caffe train \
  --solver=examples/mnist/rbm/mnist_rbm_solver.prototxt
