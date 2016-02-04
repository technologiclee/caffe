#include <numeric>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>

#include "caffe/blob.hpp"
#ifdef USE_CUDNN

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/unsupervised_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
namespace caffe {
  
/**
 * Ok, so this is getting somewhere.  But there is some good and some bad news.
 * 
 * The bad news is that the current implementation of the cudnn convolution layers are a real disaster
 * 
 * the good news is that the version in the newest caffe is a whole lot better
 * 
 * the bad news is that getting set up with that version may actually be a whole lot of work.
 * 
 * fake it till you make it; I think it should be possible to hack it together with the 
 * current version of caffe.  then next quarter I can clean it up with evan.
 * 
 * Actually, it didn't seem that hard to write the GPU forward and backward.  With a bit of good luck
 * the other stuff won't be that hard either :)
 * 
 * 
 */

__global__ void rbm_sync_conv_groups() { }

template <typename Dtype>
__global__ void SigmoidKernel(const int n, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data[index] = 1. / (1. + exp(-data[index]));
  }
}

template <typename Dtype>
__global__ void SigmoidKernel(const int n, const Dtype* bias, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) {
    data[index] = 1. / (1. + exp(-(data[index] + bias[0])));
  }
}

template <typename Dtype>
__global__ void ExpAddKernel(const int n, const Dtype* input, const Dtype alpha, Dtype* output) {
  CUDA_KERNEL_LOOP(i, n) {
    output[i] += alpha * log(1. + exp(input[i]));
  }
}

/// for each grid, calculate the probability of an element being one and sample
template <typename Dtype>
__global__ void MultinomialSampleKernel(const int n, const int pooling_size, 
                                        const int width_out, const Dtype* data_in, 
                                        const Dtype* random, Dtype* data_out) {
  int start_index;
  Dtype total_sum, running_sum, rn;
  CUDA_KERNEL_LOOP(index, n) {
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
    rn = random[index];
    for (int i = 0; i < pooling_size; i++) {
      for (int j = 0; j < pooling_size; j++) {
        int data_index = start_index + i * width_out + j;
        data_out[data_index] = 
          (Dtype)(running_sum <= total_sum * rn && total_sum * rn < (running_sum += exp(data_in[data_index])));
      }
    }
  }
}

/// for each grid, sample from the already calculated multinomial probabilites
template <typename Dtype>
__global__ void MultinomialJustSampleKernel(const int n, const int pooling_size, 
                                        const int width_out, const Dtype* data_in, 
                                        const Dtype* random, Dtype* data_out) {
  int start_index;
  Dtype running_sum, rn;
  CUDA_KERNEL_LOOP(index, n) {
    // first the how far along we are on the width
    start_index = (index % (width_out / pooling_size)) * pooling_size;
    
    // now add the height (and also go to the other batches and outputs)
    start_index += (width_out * pooling_size) * (index / (width_out / pooling_size));
    
    // now for each h_{ij} see if the random number is between the sum of all the last h_{ij} and the sum plus this one
    running_sum = 0;
    rn = random[index];
    for (int i = 0; i < pooling_size; i++) {
      for (int j = 0; j < pooling_size; j++) {
        bool in_range = running_sum <= rn && rn < (running_sum += data_in[start_index + i * width_out + j]);
        data_out[start_index + i * width_out + j] = in_range ? Dtype(1.) : Dtype(0.);
      }
    }
  }
}

/// for each grid, calculate the probability of an element being one
template <typename Dtype>
__global__ void MultinomialKernel(const int n, const int pooling_size, 
                                  const int width_out, const Dtype* data_in, 
                                  Dtype* data_out) {
  int start_index;
  Dtype total_sum;
  CUDA_KERNEL_LOOP(index, n) {
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
    for (int i = 0; i < pooling_size; i++) {
      for (int j = 0; j < pooling_size; j++) {
        int data_index = start_index + i * width_out + j;
        data_out[data_index] = exp(data_in[data_index]) / total_sum;
      }
    }
  }
}

template <typename Dtype>
__global__ void CompareKernel(const int n,
                              const Dtype* input_data,
                              const Dtype* random,
                              Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, n) {
    output_data[index] = Dtype(input_data[index] > random[index] ? 1 : 0);
  }
}

template <typename Dtype>
__global__ void CompareKernel(const int n,
                              const Dtype* start_data,
                              const Dtype* prob_data,
                              const Dtype* random,
                              const Dtype* clmp_data,
                              Dtype* output_data) {
  Dtype random_number;
  CUDA_KERNEL_LOOP(i, n) {
    random_number = Dtype(prob_data[i] > random[i] ? 1 : 0);
    output_data[i] = clmp_data[i]*start_data[i]+(1-clmp_data[i])*random_number;
  }
}

/** @brief overwrite the data with samples after calculating multinomial distibution */
template <typename Dtype>
inline void stochastic_samples(const Blob<Dtype>* blob_in,
                               Dtype* uniform_sample,
                               int pooling_size,
                               Blob<Dtype>* blob_out) {
  const Dtype* data_in = blob_in->gpu_data();
  Dtype* data_out      = blob_out->mutable_gpu_data();
  const int width_out  = blob_in->shape(3);
  const int count      = blob_in->count() / (pooling_size * pooling_size);
  
  // create some uniform samples
  caffe_gpu_rng_uniform<Dtype>(count, Dtype(0.), Dtype(1.), uniform_sample);

  // Transform this to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultinomialSampleKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, pooling_size, width_out, data_in, uniform_sample, data_out);
  CUDA_POST_KERNEL_CHECK;
}

/** @brief overwrite the data with samples with precalculated multinomial distribution */
template <typename Dtype>
inline void stochastic_samples_precalc(const Blob<Dtype>* blob_in,
                               Dtype* uniform_sample,
                               int pooling_size,
                               Blob<Dtype>* blob_out) {
  const Dtype* data_in = blob_in->gpu_data();
  Dtype* data_out      = blob_out->mutable_gpu_data();
  const int width_out  = blob_in->shape(3);
  const int count      = blob_in->count() / (pooling_size * pooling_size);
  
  // create some uniform samples
  caffe_gpu_rng_uniform<Dtype>(count, Dtype(0.), Dtype(1.), uniform_sample);

  // Transform this to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultinomialJustSampleKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, pooling_size, width_out, data_in, uniform_sample, data_out);
  CUDA_POST_KERNEL_CHECK;
}

/** @brief use probabilities from probs to create samples writen to samps */
template <typename Dtype>
inline void make_samples_from_diff(Blob<Dtype>* probs, Blob<Dtype>* samps, Dtype* uniform_sample) {
  CHECK_EQ(probs->count(), samps->count());
  const Dtype* prob_data = probs->gpu_diff();
  Dtype* samp_data = samps->mutable_gpu_data();

  const int count = probs->count();
  
  // create some uniform samples
  caffe_gpu_rng_uniform<Dtype>(count, Dtype(0.), Dtype(1.), uniform_sample);

  // Transform this to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  CompareKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, prob_data, uniform_sample, samp_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void squash(const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void squash_diff(const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, bottom[0]->mutable_gpu_diff());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void multinomial_squash(const Blob<Dtype>* blob_in, int pooling_size, Blob<Dtype>* blob_out) {
  const int width_out  = blob_in->shape(3);
  const int count      = blob_in->count() / (pooling_size * pooling_size);
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultinomialKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, pooling_size, width_out, blob_in->gpu_data(), blob_out->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (forward_is_update_) {
    Update_gpu(bottom, top);
  } else {
    vector<Blob<Dtype>*> my_bot(1), my_top(1);
    my_bot[0] = bottom[0];
    my_top[0] = top[0];
    ConvolutionLayer<Dtype>::Forward_gpu(my_bot, my_top);
    if (top.size() > 1 + num_errors_) {
      multinomial_squash(top[0], pooling_size_, top[1]);
      stochastic_samples_precalc(top[1], static_cast<Dtype*>(rng_data_->mutable_gpu_data()), this->pooling_size_, top[1]);
    }
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int g = 0; g < this->group_; g++) {
    CUDNN_CHECK(cudnnConvolutionBackwardData_v3(
          this->handle_[2*this->group_ + g],
          cudnn::dataType<Dtype>::one,
          this->filter_desc_, weight + this->weight_offset_ * g,
          this->top_descs_[0], top_data + this->top_offset_ * g,
          this->conv_descs_[0],
          this->bwd_data_algo_[0], this->workspace[2*this->group_ + g],
          this->workspace_bwd_data_sizes_[0],
          cudnn::dataType<Dtype>::zero,
          this->bottom_descs_[0], bottom_diff + this->bottom_offset_ * g));
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  rbm_sync_conv_groups<<<1, 1>>>();
  
  // add the visable bias
  if (this->visable_bias_term_) {
    const Dtype visable_bias = this->blobs_[visable_bias_index_]->cpu_data()[0];
    caffe_gpu_add_scalar(bottom[0]->count(), visable_bias, bottom_diff);
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::SampleForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<Blob<Dtype>*> my_bot(0), my_top(0);
  my_bot.push_back(bottom[0]);
  my_top.push_back(top[0]);

  CuDNNConvolutionLayer<Dtype>::Forward_gpu(my_bot, my_top);
  stochastic_samples(top[0], static_cast<Dtype*>(rng_data_->mutable_gpu_data()), pooling_size_, top[0]);
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::SampleBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> prop_down;  // dummy variable
  this->Backward_cpu(top, prop_down, bottom);
  squash_diff(bottom);
  make_samples_from_diff(bottom[0], bottom[0], static_cast<Dtype*>(rng_data_->mutable_gpu_data()));
}

template<typename Dtype>
__global__ void reduce_kernel(const int n, const int step, const Dtype* first, Dtype* data_out, const Dtype* alpha)
{
  CUDA_KERNEL_LOOP(index, n) {
    data_out[index] = *alpha * thrust::reduce(thrust::cuda::par, first + index * step, first + (index + 1) * step, Dtype(0.), thrust::plus<Dtype>());
  }
}

template<typename Dtype>
__global__ void reduce_kernel(const int n, const int step, const Dtype* first, Dtype* data_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    data_out[index] += thrust::reduce(thrust::cuda::par, first + index * step, first + (index + 1) * step, Dtype(0.), thrust::plus<Dtype>());
  }
}

template<typename Dtype>
__global__ void subtract_kernel(const int n, const int step, const Dtype* first, Dtype* data_out, const Dtype* alpha)
{
  CUDA_KERNEL_LOOP(index, n) {
    data_out[index] -= *alpha * thrust::reduce(thrust::cuda::par, first + index * step, first + (index + 1) * step, Dtype(0.), thrust::plus<Dtype>());
  }
}

template<typename Dtype>
__global__ void subtract_kernel(const int n, const int step, const Dtype* first, Dtype* data_out)
{
  CUDA_KERNEL_LOOP(index, n) {
    data_out[index] -= thrust::reduce(thrust::cuda::par, first + index * step, first + (index + 1) * step, Dtype(0.), thrust::plus<Dtype>());
  }
}

template<typename Dtype>
__global__ void mult_add(const int n, const Dtype* data_in, const Dtype alpha, Dtype* data_out) {
  CUDA_KERNEL_LOOP(index, n) {
    data_in[index] += alpha * data_out[index];
  }
}

template <typename Dtype>
void RBMCuDNNConvolutionLayer<Dtype>::Update_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  Dtype* error_vector = 0;
  if (num_errors_ > 0) {
    error_vector = top[top.size()-1]->mutable_cpu_data();
  }
  // set up the vectors that hold the hidden data
  vector<Blob<Dtype>*> hidden(0);
  hidden.push_back(top[0]);
  
  // TODO: make this a member variable, and set its size during resize
  Blob<Dtype> visable_samp(bottom[0]->shape());
  vector<Blob<Dtype>*> visable(0);
  visable.push_back(bottom[0]);
  
  CuDNNConvolutionLayer<Dtype>::Forward_gpu(visable, hidden);
  
  //during backwards pass we'll rewrite visable, so set it to a local variable
  visable[0] = &visable_samp;
  multinomial_squash(hidden[0], pooling_size_, hidden[0]);

  for (int g = 0; g < this->group_; g++) {
    // update the bias diffs with \delta b -= P(h | v_0)
    if (this->bias_term_) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(this->handle_[1*this->group_ + g],
            cudnn::dataType<Dtype>::minusone,
            this->top_descs_[0],  hidden[0]->gpu_data() + this->top_offset_ * g,
            cudnn::dataType<Dtype>::one,
            this->bias_desc_, this->blobs_[1]->mutable_gpu_diff() + this->bias_offset_ * g));
    }
    
    // update the weight diffs
    // delta w_ij -= P(H_i | v_0) * v_j^0
    // we can get away with (Dtype)1 since the weight diff is set to zero by solver
    CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(
          this->handle_[2*this->group_ + g],
          cudnn::dataType<Dtype>::minusone,
          this->bottom_descs_[0], bottom[0]->gpu_data() + this->bottom_offset_ * g,
          this->top_descs_[0],    hidden[0]->gpu_data() + this->top_offset_ * g,
          this->conv_descs_[0],
          this->bwd_filter_algo_[0], this->workspace[1*this->group_ + g],
          this->workspace_bwd_filter_sizes_[0],
          cudnn::dataType<Dtype>::one,
          this->filter_desc_, this->blobs_[0]->mutable_gpu_diff() + this->weight_offset_ * g));
  }
//  std::cout << "minus blob update = " << std::accumulate(this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->mutable_cpu_diff() + this->blobs_[1]->count(), Dtype(0.)) / this->blobs_[1]->count()  << std::endl;
  // update visable bias with \delta b -= mean(v_0)
  if (this->visable_bias_term_) {
    thrust::device_ptr<const Dtype> d_ptr(bottom[0]->gpu_data());
    Dtype update = thrust::reduce(thrust::cuda::par, d_ptr, d_ptr + bottom[0]->count());
    this->blobs_[visable_bias_index_]->mutable_cpu_diff()[0] -= update / visable[0]->count(1);
  } else {
    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    rbm_sync_conv_groups<<<1, 1>>>();
  }

  //now sample the probabilites of the hidden layer
  stochastic_samples_precalc(hidden[0], static_cast<Dtype*>(rng_data_->mutable_gpu_data()), this->pooling_size_, hidden[0]);
  
  // do backwards pass to the visable layer
  SampleBackward_gpu(hidden, visable);
  
  // calculate the reconstruction error that will be returned as the loss
  if (error_vector) {
    switch (this->layer_param_.rbm_inner_product_param().loss_measure()) {
      case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
        // copy the reconstruction from the backwards pass to the error vector
        caffe_copy(visable_samp.count(), visable_samp.cpu_diff(), error_vector);
        break;
      case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
        // The free energy error was already calculated above
        break;
      default:
        LOG(FATAL)
            << "Unknown loss measure: "
            << this->layer_param_.rbm_inner_product_param().loss_measure();
    }
  }
  
  for(int i = 0; i < num_sample_steps_for_update_ - 1; ++i) {
    SampleForward_gpu(visable, hidden);
    SampleBackward_gpu(hidden, visable);
  }
  CuDNNConvolutionLayer<Dtype>::Forward_gpu(visable, hidden);
  multinomial_squash(hidden[0], pooling_size_, hidden[0]);

  for (int g = 0; g < this->group_; g++) {
    // update the bias diffs with \delta b += P(h | v_1)
    if (this->bias_term_) {
      CUDNN_CHECK(cudnnConvolutionBackwardBias(this->handle_[1*this->group_ + g],
            cudnn::dataType<Dtype>::one,
            this->top_descs_[0],  hidden[0]->gpu_data() + this->top_offset_ * g,
            cudnn::dataType<Dtype>::one,
            this->bias_desc_, this->blobs_[1]->mutable_gpu_diff() + this->bias_offset_ * g));
    }
    
    // delta w_ij += P(H_i | v_k) * v_j^k
    CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(this->handle_[2*this->group_ + g],
          cudnn::dataType<Dtype>::one,
          this->bottom_descs_[0], visable[0]->gpu_data() + this->bottom_offset_ * g,
          this->top_descs_[0],    hidden[0]->gpu_data() + this->top_offset_ * g,
          this->conv_descs_[0],
          this->bwd_filter_algo_[0], this->workspace[1*this->group_ + g],
          this->workspace_bwd_filter_sizes_[0],
          cudnn::dataType<Dtype>::one,
          this->filter_desc_, this->blobs_[0]->mutable_gpu_diff() + this->weight_offset_ * g));
  }
  // update visable bias with \delta b += mean(v_k)
  if (this->visable_bias_term_) {
    thrust::device_ptr<const Dtype> d_ptr(visable[0]->gpu_data());
    Dtype update = thrust::reduce(thrust::cuda::par, d_ptr, d_ptr + visable[0]->count());
    // divide by the number of inputs per image, since we don't want a bigger update with bigger images
    this->blobs_[visable_bias_index_]->mutable_cpu_diff()[0] += update / visable[0]->count(1);
  } else {
    rbm_sync_conv_groups<<<1, 1>>>();
  }
}
/*
 * Ok, not there yet but I've gotten a good ways.  I guess there are two open issues:
 * 1) I probably do need a visable bias.  Just like in the paper
 * 2) I'll need to add this visable bias to the free energy calculation
 * 3) I still don't understand why the reconstruction error seems to have nothing to do with my actions
 * 
 * I need to figure out if it's first free energy then pooling or the other way around
 */


template void RBMCuDNNConvolutionLayer<float>::SampleForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMCuDNNConvolutionLayer<double>::SampleForward_gpu(
    const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);
template void RBMCuDNNConvolutionLayer<float>::SampleBackward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMCuDNNConvolutionLayer<double>::SampleBackward_gpu(
    const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);
template void RBMCuDNNConvolutionLayer<float>::Update_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMCuDNNConvolutionLayer<double>::Update_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

INSTANTIATE_LAYER_GPU_FUNCS(RBMCuDNNConvolutionLayer);

}  // namespace caffe
#endif