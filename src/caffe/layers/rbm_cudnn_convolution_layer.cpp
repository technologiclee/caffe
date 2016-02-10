#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/unsupervised_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"

namespace caffe {
  
/** @brief overwrite the data with samples */
template <typename Dtype>
inline void stochastic_samples(const Blob<Dtype>* blob_in,
                               Dtype* uniform_sample,
                               int pooling_size,
                               Blob<Dtype>* blob_out) {
  const Dtype* data_in = blob_in->cpu_data();
  Dtype* data_out = blob_out->mutable_cpu_data();
  const int width_out = blob_in->shape(3);
  const int count = blob_in->count() / (pooling_size * pooling_size);
  int start_index;
  Dtype total_sum, running_sum;
  
  // create some uniform samples
  caffe_rng_uniform<Dtype>(count, 0., 1., uniform_sample);
  for (int index = 0; index < count; ++index) {
    // first the how far along we are on the width
    start_index = (index % (width_out / pooling_size)) * pooling_size;
    
    // now add the height (and also go to the other batches and outputs)
    start_index += (width_out * pooling_size) * (index / (width_out / pooling_size));
    
    // remember that we want to have some h_{ij} be zero, so we start the sum at 1
    total_sum = 1;
    // sum over the stuff
    for (int i = 0; i < pooling_size; i++) {
      for (int j = 0; j < pooling_size; j++) {
        total_sum += exp(data_in[start_index + i * width_out + j]);
      }
    }
    // now for each h_{ij} see if the random number is between the sum of all the last h_{ij} and the sum plus this one
    running_sum = 0;
    Dtype rn = total_sum * uniform_sample[index];
    for (int i = 0; i < pooling_size; i++) {
      for (int j = 0; j < pooling_size; j++) {
        data_out[start_index + i * width_out + j] = 
          (Dtype)(running_sum <= rn && rn < (running_sum += exp(data_in[start_index + i * width_out + j])));
      }
    }
  }
}
  
/** @brief use probabilities from probs to create samples writen to samps */
template <typename Dtype>
inline void make_samples_from_diff(Blob<Dtype>* probs, Blob<Dtype>* samps) {
  int random_number;
  const Dtype* prob_data = probs->cpu_diff();
  Dtype* samp_data = samps->mutable_cpu_data();
  for (int i = 0; i < probs->count(); ++i) {
    caffe_rng_bernoulli(1, prob_data[i], &random_number);
    samp_data[i] = random_number;
  }
}

template <typename Dtype>
void squash(const vector<Blob<Dtype>*>& top) {
  Dtype* data = top[0]->mutable_cpu_data();
  for (int i = 0; i < top[0]->count(); ++i)
    data[i] = 1. / (1. + std::exp(-data[i]));
}

template <typename Dtype>
void squash_diff(const vector<Blob<Dtype>*>& bottom) {
  // do the squashing function
  Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < bottom[0]->count(); ++i) {
    bottom_data[i] = Dtype(1. / (1 + std::exp(-1 * bottom_data[i])));
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // TODO: set up a gaurd against double layer setup (also get visable_bias_term_ out of the base definition)
  vector<Blob<Dtype>*> my_bot(1), my_top(1);
  my_bot[0] = bottom[0];
  my_top[0] = top[0];
  this->visable_bias_term_ =
    this->layer_param_.rbm_inner_product_param().visable_bias_term();
  CuDNNConvolutionLayer<Dtype>::LayerSetUp(my_bot, my_top);
  //num_errors_ = this->layer_param_.rbm_inner_product_param().loss_measure_size();
  num_errors_ = int(this->layer_param_.rbm_inner_product_param().has_loss_measure());
  CHECK_LE(top.size(), 3 + num_errors_) << "top is errors plus presqaush, squash and sample";
  
  num_sample_steps_for_update_ =
    this->layer_param_.rbm_inner_product_param().sample_steps_in_update();
  pooling_size_ = this->layer_param_.rbm_convolution_param().pooling_size();
  CHECK_GE(pooling_size_, 1) << "Pooling size must be greater than zero";
  
  if (this->visable_bias_term_ && this->blobs_.size() < 1 + this->bias_term_ + this->visable_bias_term_) {
    vector<int> bias_shape(1, 1);
    this->blobs_.push_back(
      shared_ptr<Blob<Dtype> >(new Blob<Dtype>(bias_shape)));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.rbm_inner_product_param().visable_bias_filler()));
    bias_filler->Fill(this->blobs_[this->blobs_.size()-1].get());
    visable_bias_index_ = this->blobs_.size()-1;
  }
  
  // number of inputs
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));
  forward_is_update_ = this->layer_param_.rbm_convolution_param().forward_is_update();
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<Blob<Dtype>*> my_bot(1), my_top(1);
  my_bot[0] = bottom[0];
  my_top[0] = top[0];
  CuDNNConvolutionLayer<Dtype>::Reshape(my_bot, my_top);
  // shape the extra top blobs to what they should be
  for (int i = 1; i < top.size() - num_errors_; ++i) {
    top[i]->Reshape(top[0]->shape());
  }
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  CHECK_EQ(height_out % pooling_size_, 0) 
    << "output heigth (" << height_out << ") must be divisable by pooling size (" << pooling_size_ << ")";
  CHECK_EQ(width_out % pooling_size_, 0)
    << "output width (" << width_out << ") must be divisable by pooling size (" << pooling_size_ << ")";
  
  // Here, any top is the reconstruction error  
  if (num_errors_) {
    vector<int> blob_shape(2,1);
    switch(this->layer_param_.rbm_inner_product_param().loss_measure()) {
    case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
      top[top.size()-1]->ReshapeLike(*bottom[0]);
      break;
    case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
      blob_shape[0] = this->num_;  // num_ is batch size
      top[top.size()-1]->Reshape(blob_shape);
      break;
    default:
      LOG(FATAL) << "Unknown loss measure: "
        << this->layer_param_.rbm_inner_product_param().loss_measure();
    }
  }
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  if (max_count * sizeof(Dtype) > rng_data_->size()) {
    rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (forward_is_update_) {
    Update_cpu(bottom, top);
  } else {
    vector<Blob<Dtype>*> my_bot(1), my_top(1);
    my_bot[0] = bottom[0];
    my_top[0] = top[0];
    ConvolutionLayer<Dtype>::Forward_cpu(my_bot, my_top);
    if (top.size() > 1 + num_errors_) {
      for (int i = 0; i < top[0]->count(); ++i) {
        top[1]->mutable_cpu_data()[i] = Dtype(1. / (1 + std::exp(-1 * top[0]->cpu_data()[i])));
      }
      if (top.size() > 2 + num_errors_) {
        stochastic_samples(top[0], static_cast<Dtype*>(rng_data_->mutable_cpu_data()), this->pooling_size_, top[2]);
      }
    }
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // TODO: do checks on the sizes of these inputs
  const Dtype* weight   = this->blobs_[0]->cpu_data();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff    = bottom[0]->mutable_cpu_diff();
  for (int n = 0; n < this->num_; ++n) {
    this->backward_cpu_gemm(top_data + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
  }
  if(this->visable_bias_term_) {
    const Dtype visable_bias = this->blobs_[visable_bias_index_]->cpu_data()[0];
    for (int i = 0; i < bottom[0]->count(); ++i) {
      bottom_diff[i] += visable_bias;
    }
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::SampleForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<Blob<Dtype>*> my_bot(1), my_top(1);
  my_bot[0] = bottom[0];
  my_top[0] = top[0];
  ConvolutionLayer<Dtype>::Forward_cpu(my_bot, my_top);
  
  Dtype* random = static_cast<Dtype*>(rng_data_->mutable_cpu_data());
  stochastic_samples(top[0], random, this->pooling_size_, top[0]);
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::SampleBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> prop_down;  // dummy variable
  this->Backward_cpu(top, prop_down, bottom);
  if(true) {
  caffe_rng_gaussian(bottom[0]->count(), Dtype(0), Dtype(1.), static_cast<Dtype*>(bottom[0]->mutable_cpu_data()));
  caffe_axpy(bottom[0]->count(), Dtype(1.), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_data());
  } else {
  squash_diff(bottom);
  make_samples_from_diff(bottom[0], bottom[0]);
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Update_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

INSTANTIATE_CLASS(RBMCuDNNConvolutionLayer);
REGISTER_LAYER_CLASS(RBMCuDNNConvolution);

}  // namespace caffe