#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"

#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"

#include "caffe/util/device_alternate.hpp"

namespace caffe {

/** @brief overwrite the data with samples */
template <typename Dtype>
inline void make_samples(const Blob<Dtype>* input, Blob<Dtype>* samples) {
  int random_number;
  const Dtype* input_data = input->cpu_data();
  Dtype* sample_data = samples->mutable_cpu_data();

  for (int i = 0; i < input->count(); ++i) {
    caffe_rng_bernoulli(1, input_data[i], &random_number);
    sample_data[i] = (Dtype)random_number;
  }
}

/** @brief The following class provides a wrapper for a Blob which swaps the
 * data_ and diff_ arrays.
 **/
template <typename Dtype>
class SwapBlob: public Blob<Dtype> {
 public:
  explicit SwapBlob(const Blob<Dtype>* other) : Blob<Dtype>(other->shape()) {
    SetUp(other);
  }
  void SetUp(const Blob<Dtype>* other) {
    CHECK_EQ(this->count_, other->count());
    this->diff_ = other->data();
    this->data_ = other->diff();
  }
};


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
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_num_visible = bottom[0]->count(axis);
  CHECK_EQ(num_visible_, new_num_visible);
  batch_size_ = bottom[0]->count(0, axis);

  // Reshape each of the layers for the forward pass.
  connection_layer_->Reshape(bottom, pre_activation_h1_vec_);
  activation_layer_->Reshape(pre_activation_h1_vec_, mean_h1_vec_);
  sampling_layer_->Reshape(mean_h1_vec_, top);

  // The blobs for hidden to visible steps must also be reshaped.
  pre_activation_v1_vec_[0]->Reshape(bottom[0]->shape());
  mean_v1_vec_[0]->Reshape(bottom[0]->shape());

  // Ensure the size of the random number generator scratch is correct.
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  if (max_count > rng_data_->size()) {
    rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));
  }

  // Reshape the visible bias multiplier.
  // The bias multiplier is a COLUMN vector of 1's that is used to apply either
  // the hidden or the visible bias to each vector in the batch.
  // (set to M_: the size of the batch), resize the multiplier.
  // TODO: Is it required to also set it here? Check the usage of the
  // multiplier.
  if (visible_bias_term_) {
    vector<int> bias_multiplier_shape(1, this->batch_size_);
    this->bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(this->batch_size_, Dtype(1),
              this->bias_multiplier_.mutable_cpu_data());
  }

  // Reshape the error.
  if (top.size() > 1) {
    top[1]->ReshapeLike(*bottom[0]);
  }


}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    sample_h_given_v(bottom, top);
    // In unsupervised mode, run a number of gibs chains.
    if (true) {
      gibbs_hvh(bottom, top);
    }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

    if (false) {
      // For a supervised backward pass, the layer acts as a activation layer
      // on top of a connected layer.
      // TODO: Check the state of propagate down.
      activation_layer_->Backward(top, propagate_down, pre_activation_h1_vec_);
      connection_layer_->Backward(pre_activation_h1_vec_, propagate_down, bottom);
    } else {
      // Disable the update of the diffs for the weights and hidden bias.
      connection_layer_->set_param_propagate_down(0, false);
      connection_layer_->set_param_propagate_down(1, false);

      Blob<Dtype> visible_blob(bottom[0]->shape());
      vector<Blob<Dtype>*> visible(0);
      visible.push_back(&visible_blob);
      sample_v_given_h(top, visible);

      SwapBlob<Dtype> mean_v1_diff(mean_v1_vec_[0]);

      bottom[0]->CopyFrom(mean_v1_diff, true, false);

      connection_layer_->set_param_propagate_down(0, true);
      connection_layer_->set_param_propagate_down(1, true);

    }

}


template <typename Dtype>
void RBMInnerProductLayer<Dtype>::update_diffs(const int k,
    const vector<Blob<Dtype>*>& hidden_k,
    const vector<Blob<Dtype>*>& visible_k) {

  SwapBlob<Dtype> swapped_hidden_k(hidden_k[0]);
  vector<Blob<Dtype>*> hidden_vec;

  // Update the diffs for the weights and hidden bias
  vector<bool> propagate_down;
  propagate_down.push_back(false);

  if (k != 0) {
    // In order to get the summation to work out correctly, we need to multiply
    // the hidden vector by -1.
    Blob<Dtype> scaled_k;
    scaled_k.CopyFrom(*hidden_k[0], false, true);
    scaled_k.scale_data((Dtype)-1.);
    swapped_hidden_k.SetUp(&scaled_k);
  }
  hidden_vec.push_back(&swapped_hidden_k);
  connection_layer_->Backward(hidden_vec, propagate_down, visible_k);

  // Update the diffs for the visible bias
  // Update the visible bias diff (delta b -= v_k).
  if (visible_bias_term_) {
    const Dtype* v_k = visible_k[0]->cpu_data();
    Dtype* vbias_diff = this->blobs_[visible_bias_index_]->mutable_cpu_diff();
    // Gradient with respect to bias
    Dtype factor = (k == 0) ? (Dtype)(1) : (Dtype)(-1.);
    caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, num_visible_, factor,
        v_k, bias_multiplier_.cpu_data(), (Dtype)1., vbias_diff);
  }

}


template <typename Dtype>
void RBMInnerProductLayer<Dtype>::gibbs_hvh(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // TODO: make this a member variable, and set its size during resize
  Blob<Dtype> visible_blob(bottom[0]->shape());
  vector<Blob<Dtype>*> visible(0);
  visible.push_back(&visible_blob);

  // Update the diffs for k = 0, P(h|v_0)
  update_diffs(0, top, bottom);

  // Disable the update of the diffs for the weights and hidden bias.
  connection_layer_->set_param_propagate_down(0, false);
  connection_layer_->set_param_propagate_down(1, false);

  // Perform k Gibbs sampling steps.
  for (int k = 0; k < num_sample_steps_for_update_; k++) {
    // Down propagation
    sample_v_given_h(visible, top);

    caffe_copy(pre_activation_v1_vec_[0]->count(), pre_activation_v1_vec_[0]->cpu_diff(),
               top[1]->mutable_cpu_data());
    // Up propagation
    sample_h_given_v(visible, top);
  }

  // Enable the update of the diffs for the weights and hidden bias again.
  connection_layer_->set_param_propagate_down(0, true);
  connection_layer_->set_param_propagate_down(1, true);

  // Update the diffs for k, P(h|v_k)
  update_diffs(num_sample_steps_for_update_, top, visible);

  // Copy the diff (visible samples) to the bottom.
  SwapBlob<Dtype> swap_visible(&visible_blob);
  bottom[0]->CopyFrom(swap_visible, true, false);

}


template <typename Dtype>
void RBMInnerProductLayer<Dtype>::sample_h_given_v(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Do a forward pass through each of the layers.
  connection_layer_->Forward(bottom, pre_activation_h1_vec_);
  activation_layer_->Forward(pre_activation_h1_vec_, mean_h1_vec_);
  // TODO: Add the sampling layer.
  // sampling_layer_->Forward(mean_vec_, top);
  make_samples(mean_h1_vec_[0], top[0]);
}


template <typename Dtype>
void RBMInnerProductLayer<Dtype>::sample_v_given_h(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  SwapBlob<Dtype> swapped_top(top[0]);
  vector<Blob<Dtype>*> h1;
  h1.push_back(&swapped_top);

  SwapBlob<Dtype> swapped_pre_activation(pre_activation_v1_vec_[0]);
  vector<Blob<Dtype>*> pre_activation_v1;
  pre_activation_v1.push_back(&swapped_pre_activation);

  // Do a backward pass through the connection layer.
  vector<bool> propagate_down(0);
  propagate_down.push_back(true);
  connection_layer_->Backward(h1, propagate_down, pre_activation_v1);
  // Add the visible bias to the pre activation.
  if (visible_bias_term_) {
    const Dtype* vbias = this->blobs_[visible_bias_index_]->cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->batch_size_,
                          this->num_visible_, 1,
                          (Dtype)1., this->bias_multiplier_.cpu_data(),
                          vbias,
                          (Dtype)1., pre_activation_v1_vec_[0]->mutable_cpu_data());
  }

  // Do a forward pass through the activation layer.
  activation_layer_->Forward(pre_activation_v1_vec_, mean_v1_vec_);

  // TODO: Add the sampling layer.
  // Sample the mean field and store this in the bottom.
  make_samples(mean_v1_vec_[0], bottom[0]);

}


INSTANTIATE_CLASS(RBMInnerProductLayer);
REGISTER_LAYER_CLASS(RBMInnerProduct);
}  // namespace caffe
