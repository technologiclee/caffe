#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();
  if (param.forward_is_update()) {
    // do some sampling and then an update
  } else {
    // just sample forwards
    sample_h_given_v(bottom, top);
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(RBMInnerProductLayer);

}  // namespace caffe
