#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/unsupervised_layers.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename Dtype>
Dtype calculate_energy(const Blob<Dtype>& visable_bias,
                       const Blob<Dtype>& hidden_bias,
                       const Blob<Dtype>& weight_matrix,
                       const Blob<Dtype>& visable_vector,
                       const Blob<Dtype>& hidden_vector) {
  Dtype matrix_sum = caffe_cpu_dot(visable_bias.count(),
                       visable_bias.cpu_data(), visable_vector.cpu_data());

  matrix_sum += caffe_cpu_dot(hidden_bias.count(),
                       hidden_bias.cpu_data(), hidden_vector.cpu_data());

  Blob<Dtype> tmp_blob(vector<int>(1, hidden_vector.count()));

  caffe_cpu_gemv(CblasNoTrans, hidden_vector.count(), visable_vector.count(),
      (Dtype)1., weight_matrix.cpu_data(), visable_vector.cpu_data(),
      (Dtype)0., tmp_blob.mutable_cpu_data());

  matrix_sum += caffe_cpu_dot(hidden_vector.count(),
                       hidden_vector.cpu_data(), tmp_blob.cpu_data());

  // no negative due to double negation
  return exp(matrix_sum);
}

template <typename Dtype>
vector<Dtype> sample_frequency(const Blob<Dtype>& visable_bias,
                              const Blob<Dtype>& hidden_bias,
                              const Blob<Dtype>& weight_matrix,
                              const vector<shared_ptr<Blob<Dtype> > >& visable,
                              const vector<shared_ptr<Blob<Dtype> > >& hidden) {
  Dtype total = 0;
  vector<Dtype> probability(visable.size());

  for (int i = 0; i < visable.size(); i++) {
    probability[i] = 0;
    for (int j = 0; j < hidden.size(); j++) {
      probability[i] += calculate_energy(visable_bias, hidden_bias,
                            weight_matrix, *visable[i], *hidden[j]);
    }
    total += probability[i];
  }
  for (int i = 0; i < visable.size(); i++) {
    probability[i] /= total;
  }
  return probability;
}

template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > > get_combi(int blob_length) {
  const int num_combinations = pow(2, blob_length);
  vector<shared_ptr<Blob<Dtype> > > answer;
  vector<int> blob_size(4, 1);
  blob_size[2] = blob_length;
  for (int i = 0; i < num_combinations; ++i) {
    answer.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(blob_size)));
    for (int j = 0; j < blob_length; ++j) {
      answer[i]->mutable_cpu_data()[j] =
        (i/static_cast<int>(pow(2, j))) % 2 == 0;
    }
  }
  return answer;
}

template <typename Dtype>
vector<int> multinomial(int num_samples, const vector<Dtype>& probability) {
  vector<Dtype> samples(num_samples);
  // now generate a bunch of multinomial samples
  caffe_rng_uniform(num_samples, Dtype(0), Dtype(1), samples.data());
  std::sort(samples.begin(), samples.end());
  vector<int> multinomial(num_samples);
  int current_index = 0;
  Dtype cum_sum = probability[0];
  for (int i = 0; i < num_samples; ++i) {
    if (samples[i] < cum_sum || current_index == probability.size()-1) {
      multinomial[i] = current_index;
    } else {
      cum_sum += probability[++current_index];
      i--;
    }
  }
  shuffle(multinomial.begin(), multinomial.end());
  return multinomial;
}

template <typename TypeParam>
class RBMInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RBMInnerProductLayerTest()
      : blob_bottom_input_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_input_(new Blob<Dtype>()),
        blob_top_error_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_top_vec_.push_back(blob_top_input_);
    blob_top_vec_.push_back(blob_top_error_);
  }
  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete blob_top_input_;
    delete blob_top_error_;
  }

  void fill_gaussian(Blob<Dtype>* fill_me) {
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(fill_me);
  }

  void generate_layer(int num_output, int num_sample_steps) {
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(num_output);
    inner_product_param->mutable_weight_filler()->set_type("gaussian");
    RBMInnerProductParameter* rbm_inner_product_param =
        layer_param.mutable_rbm_inner_product_param();
    rbm_inner_product_param->set_visable_bias_term(true);
    rbm_inner_product_param->set_sample_steps_in_update(num_sample_steps);
    layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
  }

  void generate_weights(int num_input, int num_output) {
    // create random vectors that should be learned
    weight_matrix_.Reshape(vector<int>(1, num_input * num_output));
    this->fill_gaussian(&weight_matrix_);

    hidden_bias_.Reshape(vector<int>(1, num_output));
    this->fill_gaussian(&hidden_bias_);

    visable_bias_.Reshape(vector<int>(1, num_input));
    this->fill_gaussian(&visable_bias_);
  }

  Dtype calculate_overlap(const vector<shared_ptr<Blob<Dtype> > >& all_visable,
                          const vector<shared_ptr<Blob<Dtype> > >& all_hidden,
                          const vector<Dtype>& actual_probability) {
    // now calculate the estimated probability of everything
    vector<Dtype> est_probability(all_visable.size());
    const Blob<Dtype>& e_weight_matrix = *layer_->blobs()[0];
    const Blob<Dtype>& e_hidden_bias   = *layer_->blobs()[1];
    const Blob<Dtype>& e_visable_bias  = *layer_->blobs()[2];

    Dtype total = 0;
    for (int i = 0; i < all_visable.size(); i++) {
      est_probability[i] = 0;
      for (int j = 0; j < all_hidden.size(); j++) {
        est_probability[i] += calculate_energy(e_visable_bias, e_hidden_bias,
                              e_weight_matrix, *all_visable[i], *all_hidden[j]);
      }
      total += est_probability[i];
    }
    Dtype divergence = 0;
    for (int i = 0; i < all_visable.size(); i++) {
      est_probability[i] /= total;
      divergence += std::min(actual_probability[i], est_probability[i]);
    }
    return divergence;
  }

  Blob<Dtype> weight_matrix_;
  Blob<Dtype> hidden_bias_;
  Blob<Dtype> visable_bias_;
  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const blob_top_input_;
  Blob<Dtype>* const blob_top_error_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<RBMInnerProductLayer<Dtype> > layer_;
};

TYPED_TEST_CASE(RBMInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMInnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);
  EXPECT_EQ(this->blob_top_input_->num(), 2);
  EXPECT_EQ(this->blob_top_input_->channels(), 10);
  EXPECT_EQ(this->blob_top_input_->height(), 1);
  EXPECT_EQ(this->blob_top_input_->width(), 1);
  EXPECT_EQ(this->blob_top_error_->num(), 2);
  EXPECT_EQ(this->blob_top_error_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_->height(), 4);
  EXPECT_EQ(this->blob_top_error_->width(), 5);
}

TYPED_TEST(RBMInnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_bias_term(false);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->fill_gaussian(this->blob_top_input_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // In order to test the forward, sigmoid layer is used to squash the output.
  shared_ptr<SigmoidLayer<Dtype> > squash(
      new SigmoidLayer<Dtype>(LayerParameter()));

  Blob<Dtype> blob_squash(this->blob_top_input_->shape());
  vector<Blob<Dtype>*> blob_squash_vec;
  blob_squash_vec.push_back(&blob_squash);
  squash->Forward(this->blob_top_vec_, blob_squash_vec);

  const Dtype* test_data = blob_squash_vec[0]->cpu_data();
  for (int i = 0; i < blob_squash_vec[0]->count(); ++i) {
    EXPECT_GE(test_data[i], 0);
    EXPECT_LE(test_data[i], 1);
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestForwardSample) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  RBMInnerProductParameter* rbm_inner_product_param =
      layer_param.mutable_rbm_inner_product_param();
  rbm_inner_product_param->set_visable_bias_term(true);
  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<Blob<Dtype>*> bottom;
  bottom.push_back(this->blob_bottom_input_);
  vector<Blob<Dtype>*> top;
  top.push_back(this->blob_top_input_);
  this->fill_gaussian(this->blob_top_input_);
  layer->SampleForward(bottom, top);
  const Dtype* top_data = this->blob_top_input_->cpu_data();
  for (int i = 0; i < this->blob_top_input_->count(); ++i) {
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == 1);
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestBackward) {
  const int num_output(3);
  const int num_input(3);
  this->generate_layer(num_output, 1);
  vector<int> shape(4, 1);
  shape[0] = 2;
  shape[2] = num_input;
  this->blob_bottom_input_->Reshape(shape);
  this->blob_top_input_->Reshape(shape);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  for (int i = 0; i < this->layer_->blobs()[0]->count(); ++i) {
    this->layer_->blobs()[0]->mutable_cpu_data()[i] = 0;
  }
  this->layer_->blobs()[0]->mutable_cpu_data()[0] = 1;
  this->layer_->blobs()[0]->mutable_cpu_data()[1] = 1;
  this->layer_->blobs()[0]->mutable_cpu_data()[4] = 1;
  this->layer_->blobs()[0]->mutable_cpu_data()[8] = 1;

  this->layer_->blobs()[2]->mutable_cpu_data()[0] = 1;
  this->layer_->blobs()[2]->mutable_cpu_data()[1] = 2;
  this->layer_->blobs()[2]->mutable_cpu_data()[2] = 3;

  this->blob_top_input_->mutable_cpu_data()[0] = 8;
  this->blob_top_input_->mutable_cpu_data()[1] = 6;
  this->blob_top_input_->mutable_cpu_data()[2] = 4;

  vector<bool> p;
  this->layer_->Backward(this->blob_top_vec_, p, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_input_->cpu_diff()[0], 9);
  EXPECT_EQ(this->blob_bottom_input_->cpu_diff()[1], 8+8);
  EXPECT_EQ(this->blob_bottom_input_->cpu_diff()[2], 7);
}

TYPED_TEST(RBMInnerProductLayerTest, TestBackwardSample) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  RBMInnerProductParameter* rbm_inner_product_param =
      layer_param.mutable_rbm_inner_product_param();
  rbm_inner_product_param->set_visable_bias_term(true);
  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<Blob<Dtype>*> bottom;
  bottom.push_back(this->blob_bottom_input_);
  vector<Blob<Dtype>*> top;
  top.push_back(this->blob_top_input_);
  this->fill_gaussian(this->blob_top_input_);
  layer->SampleBackward(top, bottom);
  const Dtype* bottom_data = this->blob_bottom_input_->cpu_data();
  for (int i = 0; i < this->blob_bottom_input_->count(); ++i) {
    EXPECT_TRUE(bottom_data[i] == 0 || bottom_data[i] == 1);
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestSample) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1702);
  const int num_input(4), num_output(3), batch_size(10);
  const int num_samples(500), sample_length(10);
  this->generate_layer(num_output, 1);
  this->generate_weights(num_input, num_output);
  vector<int> input_shape(4, 1);
  input_shape[0] = batch_size;
  input_shape[2] = num_input;
  this->blob_bottom_input_->Reshape(input_shape);
  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(num_input);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(num_output);
  vector<Dtype> probability(all_visable.size());
  map<string, int> vector_to_index;

  // see how probabable different samples are
  for (int i = 0; i < all_visable.size(); i++) {
    probability[i] = 0;
    ostringstream stuff;
    for (int j = 0; j < all_visable[i]->count(); ++j) {
      stuff << "_" << all_visable[i]->cpu_data()[j];
    }
    vector_to_index[stuff.str()] = i;
  }

  // set up the layer
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // copy random weights to the layer
  caffe_copy(this->weight_matrix_.count(), this->weight_matrix_.cpu_data(),
             this->layer_->blobs()[0]->mutable_cpu_data());
  caffe_copy(this->hidden_bias_.count(), this->hidden_bias_.cpu_data(),
             this->layer_->blobs()[1]->mutable_cpu_data());
  caffe_copy(this->visable_bias_.count(), this->visable_bias_.cpu_data(),
             this->layer_->blobs()[2]->mutable_cpu_data());

  // generate samples from the layer
  for (int i = 0; i < num_samples / batch_size; ++i) {
    // this->blob_bottom_vec_[0] = all_visable[multi[i]].get();
    caffe_set(this->blob_bottom_vec_[0]->count(), Dtype(0.),
              this->blob_bottom_vec_[0]->mutable_cpu_data());
    for (int j = 0; j < sample_length; ++j) {
      this->layer_->SampleForward(this->blob_bottom_vec_, this->blob_top_vec_);
      this->layer_->SampleBackward(this->blob_top_vec_, this->blob_bottom_vec_);
    }

    // see how often we get different samples
    for (int k = 0; k < batch_size; ++k) {
      ostringstream stuff;
      for (int j = 0; j < this->blob_bottom_input_->count(1); ++j) {
        stuff << "_" << this->blob_bottom_input_->cpu_data()[k*num_input + j];
      }
      probability[vector_to_index[stuff.str()]] += Dtype(1.0) / num_samples;
    }
  }

  // make sure the divergence between exact and estimated is low
  EXPECT_GE(this->calculate_overlap(all_visable, all_hidden, probability), .9);
}

// If the weight starts off in the right spot, the update is zero on average
TYPED_TEST(RBMInnerProductLayerTest, TestZeroUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1701);
  const int num_input(4), num_output(3), batch_size(10);
  this->generate_layer(num_output, 10);
  this->generate_weights(num_input, num_output);
  vector<int> input_shape(4, 1);
  input_shape[0] = batch_size;
  input_shape[2] = num_input;
  this->blob_bottom_input_->Reshape(input_shape);

  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(num_input);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(num_output);
  vector<Dtype> probability = sample_frequency(this->visable_bias_,
                                  this->hidden_bias_, this->weight_matrix_,
                                  all_visable, all_hidden);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // set the weights of the RBM to the weights that generated the data
  caffe_copy(this->weight_matrix_.count(), this->weight_matrix_.cpu_data(),
             this->layer_->blobs()[0]->mutable_cpu_data());
  caffe_copy(this->hidden_bias_.count(), this->hidden_bias_.cpu_data(),
             this->layer_->blobs()[1]->mutable_cpu_data());
  caffe_copy(this->visable_bias_.count(), this->visable_bias_.cpu_data(),
             this->layer_->blobs()[2]->mutable_cpu_data());

  // set the weight diffs to zero
  for (int j = 0; j < 3; ++j) {
    shared_ptr<Blob<Dtype> > blob = this->layer_->blobs()[j];
    caffe_set(blob->count(), Dtype(0.), blob->mutable_cpu_diff());
  }

  const int num_samples = 500;
  vector<int> multi = multinomial(num_samples, probability);

  // Do a bunch of updates, adding the update to the diff
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int i = 0; i < num_samples / batch_size; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        caffe_copy(num_input, all_visable[multi[i*batch_size + j]]->cpu_data(),
                   this->blob_bottom_input_->mutable_cpu_data() + j*num_input);
      }
      this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int i = 0; i < num_samples / batch_size; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        caffe_copy(num_input, all_visable[multi[i*batch_size + j]]->gpu_data(),
                   this->blob_bottom_input_->mutable_gpu_data() + j*num_input);
      }
      this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
  // make sure that the average diffs are not too large
  for (int j = 0; j < 3; ++j) {
    shared_ptr<Blob<Dtype> > blob = this->layer_->blobs()[j];
    for (int i = 0; i < blob->count(); ++i) {
      EXPECT_GE(blob->cpu_diff()[i] / num_samples, -0.05);
      EXPECT_LE(blob->cpu_diff()[i] / num_samples,  0.05);
    }
  }
}

// Test if a randomly initialized RBM converges to some local minimum
TYPED_TEST(RBMInnerProductLayerTest, TestLocalMinimum) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1702);
  const int num_input(4), num_output(3), batch_size(10);
  this->generate_layer(num_output, 20);
  this->generate_weights(num_input, num_output);
  vector<int> input_shape(4, 1);
  input_shape[2] = num_input;
  this->blob_bottom_input_->Reshape(input_shape);

  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(num_input);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(num_output);
  Dtype total = 0;
  Dtype learning_rate = .02 * batch_size;
  const int num_samples = 500;
  vector<Dtype> probability = sample_frequency(this->visable_bias_,
                                  this->hidden_bias_, this->weight_matrix_,
                                  all_visable, all_hidden);

  // create a random sample of these visable vectors
  vector<int> multi = multinomial(num_samples, probability);

  // feed the visable vectors throgh the network, updating the weights
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int i = 0; i < num_samples / batch_size; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        caffe_copy(num_input, all_visable[multi[i*batch_size + j]]->cpu_data(),
                   this->blob_bottom_input_->mutable_cpu_data() + j*num_input);
      }
      for (int j = 0; j < 3; j++) {
        caffe_set(this->layer_->blobs()[j]->count(), Dtype(0.),
                  this->layer_->blobs()[j]->mutable_cpu_diff());
      }
      this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      for (int j = 0; j < 3; ++j) {
        shared_ptr<Blob<Dtype> > blobs = this->layer_->blobs()[j];
        caffe_axpy(blobs->count(), Dtype(learning_rate),
                   blobs->cpu_diff(), blobs->mutable_cpu_data());
      }
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int i = 0; i < num_samples / batch_size; ++i) {
      for (int j = 0; j < batch_size; ++j) {
        caffe_copy(num_input, all_visable[multi[i*batch_size + j]]->gpu_data(),
                   this->blob_bottom_input_->mutable_gpu_data() + j*num_input);
      }
      for (int j = 0; j < 3; j++)
        caffe_gpu_set(this->layer_->blobs()[j]->count(),
                      Dtype(0.), this->layer_->blobs()[j]->mutable_gpu_diff());
      this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      for (int j = 0; j < 3; ++j) {
        shared_ptr<Blob<Dtype> > blobs = this->layer_->blobs()[j];
        caffe_gpu_axpy(blobs->count(), Dtype(learning_rate),
                       blobs->gpu_diff(), blobs->mutable_gpu_data());
      }
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }

  // calculate the numerical gradient of the TV where the RBM stopped
  const Blob<Dtype>& e_weight_matrix = *this->layer_->blobs()[0];
  const Blob<Dtype>& e_hidden_bias   = *this->layer_->blobs()[1];
  const Blob<Dtype>& e_visable_bias  = *this->layer_->blobs()[2];

  vector<Dtype> ground_probability = sample_frequency(e_visable_bias,
                                      e_hidden_bias, e_weight_matrix,
                                      all_visable, all_hidden);

  // see if learning is in a local minimum, calculate overlap at learned point
  Dtype learned_overlap = 0;
  for (int j = 0; j < probability.size(); ++j) {
    learned_overlap += std::min(probability[j], ground_probability[j]);
  }
  // For every possible direction, perturb the weight and ensure the new
  // solution is not better much better
  const Dtype epsi = 0.001;
  for (int k = 0; k < 3; ++k) {
    Blob<Dtype>& blob_to_test = *this->layer_->blobs()[k];
    for (int i = 0; i < blob_to_test.count(); ++i) {
        blob_to_test.mutable_cpu_data()[i] += epsi;
        vector<Dtype> new_probability = sample_frequency(e_visable_bias,
                                        e_hidden_bias, e_weight_matrix,
                                        all_visable, all_hidden);
      Dtype new_overlap = 0;
      for (int j = 0; j < probability.size(); ++j) {
        new_overlap += std::min(probability[j], new_probability[j]);
      }
      EXPECT_GE(learned_overlap, new_overlap - .0001 * epsi);
      blob_to_test.mutable_cpu_data()[i] -= epsi;
    }
  }
}

}  //  namespace caffe
