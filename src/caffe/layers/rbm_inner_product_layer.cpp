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
  bool skip_init = (this->blobs_.size() > 0);
  // The number of hidden units should be greater than zero.
  const int num_output = this->layer_param_.inner_product_param().num_output();
  CHECK_GT(num_output, 0);

  // The number of hidden units is determined from the output of the inner
  num_hidden_ = num_output;
  // The number of visible units is determined from the bottom blob.
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  num_visible_ = bottom[0]->count(axis);
  batch_size_ = bottom[0]->count(0, axis);

  visible_bias_term_ =
      this->layer_param_.rbm_inner_product_param().visible_bias_term();
  num_sample_steps_for_update_ =
      this->layer_param_.rbm_inner_product_param().sample_steps_in_update();

  pre_activation_h1_vec_.clear();
  mean_h1_vec_.clear();
  sample_h1_vec_.clear();
  pre_activation_v1_vec_.clear();
  mean_v1_vec_.clear();
  sample_v1_vec_.clear();

  // Add a blob for the pre_activation_values
  pre_activation_h1_vec_.push_back(new Blob<Dtype>());
  pre_activation_v1_vec_.push_back(new Blob<Dtype>());
  // Add a blob for the mean values
  mean_h1_vec_.push_back(new Blob<Dtype>());
  mean_v1_vec_.push_back(new Blob<Dtype>());
  // Add a blob for the hidden samples
  sample_h1_vec_.push_back(new Blob<Dtype>());

  // Set up the layers.
  // Connection layer: Inner product.
  connection_layer_.reset(new InnerProductLayer<Dtype>(this->layer_param_));
  connection_layer_->SetUp(bottom, pre_activation_h1_vec_);

  // Activation layer: Sigmoid.
  activation_layer_.reset(new SigmoidLayer<Dtype>(this->layer_param_));
  activation_layer_->SetUp(pre_activation_h1_vec_, mean_h1_vec_);

  // TODO: Add a propper sampling layer which implements bernoulli sampling.
  // Sampling layer:
  sampling_layer_.reset(new ThresholdLayer<Dtype>(this->layer_param_));
  sampling_layer_->SetUp(mean_h1_vec_, sample_h1_vec_);
  // TODO: This should be part of the sampling layer.
  // Initialised the RNG storage for the Bernouli sampling.
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));

  // Set up the visible bias.
  if (skip_init) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Add the blobs from the connection layer to the rbm blobs:
    vector<shared_ptr<Blob<Dtype> > >& connection_blobs = connection_layer_->blobs();
    this->blobs_.resize(connection_blobs.size());
    for (int i = 0; i < connection_blobs.size(); ++i) {
      this->blobs_[i] = connection_blobs[i];
    }
    // Add the blob for the visible bias if required.
    if (visible_bias_term_) {
      visible_bias_index_ = this->blobs_.size();
      vector<int> bias_shape(1, this->num_visible_);
      this->blobs_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>(bias_shape)));
      // Fill the visible bias.
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.rbm_inner_product_param().visible_bias_filler()));
      bias_filler->Fill(this->blobs_[visible_bias_index_].get());
    }
  }
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
