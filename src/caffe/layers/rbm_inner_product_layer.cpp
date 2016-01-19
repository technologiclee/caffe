#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/unsupervised_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/** @brief overwrite the data with samples */
template <typename Dtype>
inline void make_samples(Blob<Dtype>* to_sample) {
  int random_number;
  Dtype* blob_data = to_sample->mutable_cpu_data();
  for (int i = 0; i < to_sample->count(); ++i) {
    caffe_rng_bernoulli(1, blob_data[i], &random_number);
    blob_data[i] = random_number;
  }
}

/** @brief overwrite the data with samples, clamp with clamps */
template <typename Dtype>
inline void make_samples(Blob<Dtype>* to_sample, Blob<Dtype>* clamps) {
  int random_number;
  Dtype* blob_data = to_sample->mutable_cpu_data();
  const Dtype* clmp_data = clamps->cpu_data();
  const Dtype* start_data = clamps->cpu_diff();
  for (int i = 0; i < to_sample->count(); ++i) {
    caffe_rng_bernoulli(1, blob_data[i], &random_number);
    blob_data[i] =
        clmp_data[i] * start_data[i] + (1 - clmp_data[i]) * random_number;
  }
}

/** @brief use probabilities from probs to create samples writen to samps */
template <typename Dtype>
inline void make_samples_from_diff(Blob<Dtype>* probs, Blob<Dtype>* samps) {
  CHECK_EQ(probs->count(), samps->count());
  int random_number;
  const Dtype* prob_data = probs->cpu_diff();
  Dtype* samp_data = samps->mutable_cpu_data();
  for (int i = 0; i < probs->count(); ++i) {
    caffe_rng_bernoulli(1, prob_data[i], &random_number);
    samp_data[i] = random_number;
  }
}

/**
 * @brief use probabilities from probs to create samples writen to samps
 *        and clamp with the third blob
 */
template <typename Dtype>
inline void make_samples_from_diff(Blob<Dtype>* probs, Blob<Dtype>* samps,
                                   Blob<Dtype>* clamps) {
  CHECK_EQ(probs->count(), samps->count());
  CHECK_EQ(probs->count(), clamps->count());
  int random_number;
  const Dtype* prob_data = probs->cpu_diff();
  const Dtype* clmp_data = clamps->cpu_data();
  Dtype* samp_data = samps->mutable_cpu_data();

  for (int i = 0; i < probs->count(); ++i) {
    caffe_rng_bernoulli(1, prob_data[i], &random_number);
    samp_data[i] =
        clmp_data[i] * samp_data[i] + (1 - clmp_data[i]) * random_number;
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  bool skip_init = (this->blobs_.size() > 0);
  InnerProductLayer<Dtype>::LayerSetUp(bottom, top);
  visable_bias_term_ =
      this->layer_param_.rbm_inner_product_param().visable_bias_term();
  num_sample_steps_for_update_ =
      this->layer_param_.rbm_inner_product_param().sample_steps_in_update();

  // Check if we need to set up the weights
  if (skip_init) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (visable_bias_term_) {
      vector<int> bias_shape(1, this->K_);
      this->blobs_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>(bias_shape)));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.rbm_inner_product_param().visable_bias_filler()));
      Dtype* stuff =
          this->blobs_[this->blobs_.size() - 1].get()->mutable_cpu_data();
      CHECK(stuff);
      bias_filler->Fill(this->blobs_[this->blobs_.size() - 1].get());
      visable_bias_index_ = this->blobs_.size() - 1;
    }
  }
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  if (bottom.size() > 1) {
    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
          << "Data and clamp inputs must have the same dimensions";
    }
  }
  InnerProductLayer<Dtype>::Reshape(bottom, top);
  if (top.size() > 1) {
    if (this->layer_param_.rbm_inner_product_param().second_top_is_loss()) {
      vector<int> blob_shape(2, 1);
      switch (this->layer_param_.rbm_inner_product_param().loss_measure()) {
        case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
          top[1]->ReshapeLike(*bottom[0]);
          break;
        case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
          blob_shape[0] = this->M_;  // M_ is batch size
          top[1]->Reshape(blob_shape);
          break;
        default:
          LOG(FATAL)
              << "Unknown loss measure: "
              << this->layer_param_.rbm_inner_product_param().loss_measure();
      }
      CHECK_LE(top.size(), 2)
          << "if the second_top_is_loss, you can't have third top";
    } else {
      for (int i = 0; i < top[0]->num_axes(); ++i) {
        CHECK_EQ(top[0]->shape(i), top[1]->shape(i))
            << "hidden output and clamp inputs must have the same dimensions";
      }
    }
  }
  if (top.size() > 2) {
    vector<int> blob_shape(2, 1);
    switch (this->layer_param_.rbm_inner_product_param().loss_measure()) {
      case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
        top[2]->ReshapeLike(*bottom[0]);
        break;
      case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
        blob_shape[0] = bottom[0]->shape(0);
        top[2]->Reshape(blob_shape);
        break;
      default:
        LOG(FATAL)
            << "Unknown loss measure: "
            << this->layer_param_.rbm_inner_product_param().loss_measure();
    }
  }
  const int max_count = std::max(bottom[0]->count(), top[0]->count());
  if (max_count > rng_data_->size()) {
    rng_data_.reset(new SyncedMemory(max_count * sizeof(Dtype)));
  }
  if (visable_bias_term_ && (this->bias_multiplier_.num_axes() < 2 ||
                             this->bias_multiplier_.shape(1) < this->N_)) {
    // we use this bias_multiplier_ to multiply both the hidden and vis bais
    vector<int> bias_shape(1, this->N_);
    this->bias_multiplier_.Reshape(bias_shape);
    caffe_set(this->N_, Dtype(1), this->bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void squash(const vector<Blob<Dtype>*>& top) {
  // do the squashing function
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < top[0]->count(); ++i)
    top_data[i] = 1.0 / (1 + std::exp(-1 * top_data[i]));
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Update_cpu(bottom, top);
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::SampleForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (top.size() > 1) {
    CHECK_EQ(top[0]->count(), top[1]->count())
        << "data and clamps must be same size";
    // copy the data to the clamp's diffs, sort of a hack but OK
    const int N = top[0]->count();
    caffe_copy(N, top[0]->cpu_data(), top[1]->mutable_cpu_diff());
  }
  InnerProductLayer<Dtype>::Forward_cpu(bottom, top);
  squash(top);
  if (top.size() == 1) {
    make_samples(top[0]);
  } else {
    make_samples(top[0], top[1]);
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_cpu(
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
  Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->K_,
                        this->N_, (Dtype)1., top_data, weight, (Dtype)0.,
                        bottom_data);

  // add the bias
  if (visable_bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->M_, this->K_, 1,
                          (Dtype)1., this->bias_multiplier_.cpu_data(),
                          this->blobs_[visable_bias_index_]->cpu_data(),
                          (Dtype)1., bottom_data);
  }

  // do the squashing function
  for (int i = 0; i < bottom[0]->count(); ++i) {
    bottom_data[i] = Dtype(1. / (1 + std::exp(-1 * bottom_data[i])));
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::SampleBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  vector<bool> prop_down(top.size(), false);
  this->Backward_cpu(top, prop_down, bottom);
  if (bottom.size() == 1) {
    make_samples_from_diff(bottom[0], bottom[0]);
  } else {
    make_samples_from_diff(bottom[0], bottom[0], bottom[1]);
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Update_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  vector<bool> prop_down(top.size(), false);
  Dtype* error_vector = 0;
  if (top.size() == 3) {
    error_vector = top[2]->mutable_cpu_data();
  } else {
    if (this->layer_param_.rbm_inner_product_param().second_top_is_loss()) {
      error_vector = top[1]->mutable_cpu_data();
    }
  }

  // set up the vectors that hold the hidden data
  vector<Blob<Dtype>*> hidden(0);
  hidden.push_back(top[0]);

  // TODO: make this a member variable, and set its size during resize
  Blob<Dtype> visable_samp(bottom[0]->shape());
  vector<Blob<Dtype>*> visable(0);
  visable.push_back(&visable_samp);

  // In order to calculate the free energy, we need to do the forward pass
  // without squashing.
  InnerProductLayer<Dtype>::Forward_cpu(bottom, hidden);
  if (error_vector &&
      this->layer_param_.rbm_inner_product_param().loss_measure() ==
          RBMInnerProductParameter_LossMeasure_FREE_ENERGY) {
    if (visable_bias_term_) {
      caffe_cpu_gemv(CblasNoTrans, this->M_, this->K_, Dtype(-1.),
                     bottom[0]->cpu_data(),
                     this->blobs_[visable_bias_index_]->cpu_data(), Dtype(0.),
                     error_vector);
    } else {
      caffe_set(this->M_, Dtype(0.), error_vector);
    }
    // Take the exponential function of hidden (but not yet squashed) values
    const Dtype* hidden_data = hidden[0]->cpu_data();
    Dtype* exp_data = static_cast<Dtype*>(rng_data_->mutable_cpu_data());
    for (int i = 0; i < hidden[0]->count(); ++i) {
      exp_data[i] = std::log(1 + std::exp(hidden_data[i]));
    }
    // Now multiply this by a one vector and add to error
    caffe_cpu_gemv(CblasNoTrans, this->M_, this->N_, Dtype(-1.), exp_data,
                   this->bias_multiplier_.cpu_data(), Dtype(1.), error_vector);
  }
  // now do the squashing function
  squash(hidden);

  // update the weight diffs
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  // delta w_ij -= P(H_i | v_0) * v_j^0
  // we can get away with (Dtype)1 since the weight diff is set to zero by
  // solver
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->N_, this->K_, this->M_,
                        Dtype(-1.), hidden[0]->cpu_data(),
                        bottom[0]->cpu_data(), Dtype(1.), weight_diff);

  // update the bias diffs with \delta b -= P(h | v_0)
  if (this->bias_term_) {
    Dtype* h_bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // should be something ike this
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, Dtype(-1. / this->M_),
                          hidden[0]->cpu_data(),
                          this->bias_multiplier_.cpu_data(), Dtype(1.),
                          h_bias_diff);
  }

  // update bias diffs with \delta b -= v_0
  if (visable_bias_term_) {
    Dtype* v_bias_diff = this->blobs_[visable_bias_index_]->mutable_cpu_diff();

    // should be something ike this
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->K_, Dtype(-1. / this->M_),
                          bottom[0]->cpu_data(),
                          this->bias_multiplier_.cpu_data(), Dtype(1.),
                          v_bias_diff);
  }
  // now sample the probabilites of the hidden layer
  make_samples(hidden[0]);

  // do backwards pass to the visable layer
  Backward_cpu(hidden, prop_down, visable);

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
  make_samples_from_diff(visable[0], visable[0]);

  for (int i = 0; i < num_sample_steps_for_update_ - 1; ++i) {
    SampleForward_cpu(visable, hidden);
    SampleBackward_cpu(hidden, visable);
  }
  InnerProductLayer<Dtype>::Forward_cpu(visable, hidden);
  squash(hidden);

  // delta w_ij += P(H_i | v_k) * v_j^k
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, this->N_, this->K_, this->M_,
                        Dtype(1.), hidden[0]->cpu_data(),
                        visable[0]->cpu_data(), Dtype(1.), weight_diff);

  // update the bias diffs with \delta b += h_k
  if (this->bias_term_) {
    Dtype* h_bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // should be something ike this
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->N_, Dtype(1. / this->M_),
                          hidden[0]->cpu_data(),
                          this->bias_multiplier_.cpu_data(), Dtype(1.),
                          h_bias_diff);
  }

  // update the bias diffs with \delta b = v_0 - v_k
  if (visable_bias_term_) {
    Dtype* v_bias_diff = this->blobs_[visable_bias_index_]->mutable_cpu_diff();

    // should be something ike this
    caffe_cpu_gemv<Dtype>(CblasTrans, this->M_, this->K_, Dtype(1. / this->M_),
                          visable[0]->cpu_data(),
                          this->bias_multiplier_.cpu_data(), Dtype(1.),
                          v_bias_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(RBMInnerProductLayer);
STUB_GPU_FORWARD(RBMInnerProductLayer, SampleForward);
STUB_GPU_FORWARD(RBMInnerProductLayer, Update);
STUB_GPU_FORWARD(RBMInnerProductLayer, SampleBackward);
#endif

INSTANTIATE_CLASS(RBMInnerProductLayer);
REGISTER_LAYER_CLASS(RBMInnerProduct);

}  // namespace caffe
