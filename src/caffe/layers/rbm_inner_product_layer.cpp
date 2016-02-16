#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"

#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"

#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(RBMInnerProductLayer);
#endif

INSTANTIATE_CLASS(RBMInnerProductLayer);
REGISTER_LAYER_CLASS(RBMInnerProduct);

}  // namespace caffe
