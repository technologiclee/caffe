#include <vector>

#include "caffe/filler.hpp"
#include "caffe/unsupervised_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

INSTANTIATE_LAYER_GPU_FUNCS(RBMInnerProductLayer);

template <typename Dtype>
__global__ void SigmoidKernel(const int n, Dtype* data) {
  CUDA_KERNEL_LOOP(index, n) { data[index] = 1. / (1. + exp(-data[index])); }
}

template <typename Dtype>
__global__ void CompareKernel(const int n, const Dtype* input_data,
                              const Dtype* random, Dtype* output_data) {
  CUDA_KERNEL_LOOP(index, n) {
    output_data[index] = Dtype(input_data[index] > random[index] ? 1 : 0);
  }
}

template <typename Dtype>
__global__ void CompareKernel(const int n, const Dtype* start_data,
                              const Dtype* prob_data, const Dtype* random,
                              const Dtype* clmp_data, Dtype* output_data) {
  Dtype random_number;
  CUDA_KERNEL_LOOP(i, n) {
    random_number = Dtype(prob_data[i] > random[i] ? 1 : 0);
    output_data[i] =
        clmp_data[i] * start_data[i] + (1 - clmp_data[i]) * random_number;
  }
}

template <typename Dtype>
__global__ void ExpAddKernel(const int n, const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(i, n) { output[i] = log(1. + exp(input[i])); }
}

/** @brief overwrite the data with samples */
template <typename Dtype>
inline void make_samples(Blob<Dtype>* to_sample, Dtype* uniform_sample) {
  Dtype* blob_data = to_sample->mutable_gpu_data();
  const int count = to_sample->count();

  // create some uniform samples
  caffe_gpu_rng_uniform<Dtype>(count, 0., 1., uniform_sample);

  // Transform this to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  CompareKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, blob_data, uniform_sample, blob_data);
  CUDA_POST_KERNEL_CHECK;
}

/** @brief use probabilities from probs to create samples writen to samps */
template <typename Dtype>
inline void make_samples_from_diff(Blob<Dtype>* probs, Blob<Dtype>* samps,
                                   Dtype* uniform_sample) {
  CHECK_EQ(probs->count(), samps->count());
  const Dtype* prob_data = probs->gpu_diff();
  Dtype* samp_data = samps->mutable_gpu_data();

  const int count = probs->count();

  // create some uniform samples
  caffe_gpu_rng_uniform<Dtype>(count, 0., 1., uniform_sample);

  // Transform this to zeros and ones
  // NOLINT_NEXT_LINE(whitespace/operators)
  CompareKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, prob_data, uniform_sample, samp_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void squash(const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Update_gpu(bottom, top);
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::SampleForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  InnerProductLayer<Dtype>::Forward_gpu(bottom, top);
  squash(top);
  make_samples(top[0], static_cast<Dtype*>(rng_data_->mutable_gpu_data()));
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK_EQ(bottom[0]->shape(0), top[0]->shape(0))
      << "Bottom and top must have the same batch size";
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  CHECK_EQ(top[0]->count(axis), this->N_)
      << "Top blob c*h*w must equal num_output";
  CHECK_EQ(bottom[0]->count(axis), this->K_)
      << "Bottom blob c*h*w must equal K_";
  Dtype* bottom_data = bottom[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->K_,
                        this->N_, (Dtype)1., top_data, weight, (Dtype)0.,
                        bottom_data);

  // add the bias
  if (visable_bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->K_, 1,
                          (Dtype)1., this->bias_multiplier_.gpu_data(),
                          this->blobs_[visable_bias_index_]->gpu_data(),
                          (Dtype)1., bottom_data);
  }

  // do the squashing function
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidKernel<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, bottom_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::SampleBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> prop_down(top.size(), false);
  this->Backward_gpu(top, prop_down, bottom);
  make_samples_from_diff(bottom[0], bottom[0],
                         static_cast<Dtype*>(rng_data_->mutable_gpu_data()));
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Update_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  vector<bool> prop_down(top.size(), false);
  Dtype* error_vector = 0;
  if (top.size() > 1) {
    error_vector = top[1]->mutable_gpu_data();
  }

  // set up the vectors that hold the hidden data
  vector<Blob<Dtype>*> hidden(0);
  hidden.push_back(top[0]);

  // TODO: make this a member variable, and set its size during resize
  Blob<Dtype> visable_samp(bottom[0]->shape());
  vector<Blob<Dtype>*> visable(0);
  visable.push_back(&visable_samp);

  // Do the forward pass without squashing, so that we can calculate free energy
  InnerProductLayer<Dtype>::Forward_gpu(bottom, hidden);
  if (error_vector &&
      this->layer_param_.rbm_inner_product_param().loss_measure() ==
          RBMInnerProductParameter_LossMeasure_FREE_ENERGY) {
    error_vector = top[1]->mutable_gpu_data();
    if (visable_bias_term_) {
      caffe_gpu_gemv(CblasNoTrans, this->M_, this->K_, Dtype(-1.),
                     bottom[0]->gpu_data(),
                     this->blobs_[visable_bias_index_]->gpu_data(), Dtype(0.),
                     error_vector);
    } else {
      caffe_gpu_set(this->M_, Dtype(0.), error_vector);
    }
    // Take the exponential function of hidden (but not yet squashed) values
    const Dtype* hidden_data = hidden[0]->gpu_data();
    Dtype* exp_data = static_cast<Dtype*>(rng_data_->mutable_gpu_data());
    int grid_size = CAFFE_GET_BLOCKS(hidden[0]->count());
    ExpAddKernel<Dtype> << <grid_size, CAFFE_CUDA_NUM_THREADS>>>
        (hidden[0]->count(), hidden_data, exp_data);
    CUDA_POST_KERNEL_CHECK;
    // Now multiply this by a one vector and add to error
    error_vector = top[1]->mutable_gpu_data();
    caffe_gpu_gemv(CblasNoTrans, this->M_, this->N_, Dtype(-1.), exp_data,
                   this->bias_multiplier_.gpu_data(), Dtype(1.), error_vector);
  }
  // now do the squashing function
  squash(hidden);

  // update the weight diffs
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  // delta w_ij -= P(H_i | v_0) * v_j^0
  // we can get away with (Dtype)1 since the weight diff is set to zero by
  // solver
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->N_, this->K_, this->M_,
                        Dtype(-1.), hidden[0]->gpu_data(),
                        bottom[0]->gpu_data(), Dtype(1.), weight_diff);

  // update the bias diffs with \delta b -= P(h | v_0)
  if (this->bias_term_) {
    Dtype* h_bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // should be something ike this
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, Dtype(-1. / this->M_),
                          hidden[0]->gpu_data(),
                          this->bias_multiplier_.gpu_data(), Dtype(1.),
                          h_bias_diff);
  }

  // update bias diffs with \delta b -= v_0
  if (visable_bias_term_) {
    Dtype* v_bias_diff = this->blobs_[visable_bias_index_]->mutable_gpu_diff();

    // should be something ike this
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->K_, Dtype(-1. / this->M_),
                          bottom[0]->gpu_data(),
                          this->bias_multiplier_.gpu_data(), Dtype(1.),
                          v_bias_diff);
  }
  // now sample the probabilites of the hidden layer
  make_samples(hidden[0], static_cast<Dtype*>(rng_data_->mutable_gpu_data()));

  // do backwards pass to the visable layer
  Backward_gpu(hidden, prop_down, visable);

  // calculate the reconstruction error that will be returned as the loss
  if (error_vector) {
    switch (this->layer_param_.rbm_inner_product_param().loss_measure()) {
      case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
        // copy the reconstruction from the backwards pass to the error vector
        caffe_copy(visable_samp.count(), visable_samp.gpu_diff(), error_vector);
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
  make_samples_from_diff(visable[0], visable[0],
                         static_cast<Dtype*>(rng_data_->mutable_gpu_data()));

  for (int i = 0; i < num_sample_steps_for_update_ - 1; ++i) {
    SampleForward_gpu(visable, hidden);
    SampleBackward_gpu(hidden, visable);
  }
  InnerProductLayer<Dtype>::Forward_gpu(visable, hidden);
  squash(hidden);

  // delta w_ij += P(H_i | v_k) * v_j^k
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->N_, this->K_, this->M_,
                        Dtype(1.), hidden[0]->gpu_data(),
                        visable[0]->gpu_data(), Dtype(1.), weight_diff);

  // update the bias diffs with \delta b += h_k
  if (this->bias_term_) {
    Dtype* h_bias_diff = this->blobs_[1]->mutable_gpu_diff();
    // should be something ike this
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, Dtype(1. / this->M_),
                          hidden[0]->gpu_data(),
                          this->bias_multiplier_.gpu_data(), Dtype(1.),
                          h_bias_diff);
  }

  // update the bias diffs with \delta b = v_0 - v_k
  if (visable_bias_term_) {
    Dtype* v_bias_diff = this->blobs_[visable_bias_index_]->mutable_gpu_diff();

    // should be something ike this
    caffe_gpu_gemv<Dtype>(CblasTrans, this->M_, this->K_, Dtype(1. / this->M_),
                          visable[0]->gpu_data(),
                          this->bias_multiplier_.gpu_data(), Dtype(1.),
                          v_bias_diff);
  }
}

template void RBMInnerProductLayer<float>::SampleForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMInnerProductLayer<double>::SampleForward_gpu(
    const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);
template void RBMInnerProductLayer<float>::SampleBackward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMInnerProductLayer<double>::SampleBackward_gpu(
    const vector<Blob<double>*>& top, const vector<Blob<double>*>& bottom);
template void RBMInnerProductLayer<float>::Update_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void RBMInnerProductLayer<double>::Update_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

}  // namespace caffe
