#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
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

void update_state(vector<int>& state_vector, int max_value) {
  for (int i = 0; i < state_vector.size(); ++i) {
    if (state_vector[i] == max_value) {
      state_vector[i] = 0;
    } else {
      state_vector[i]++;
      break;
    }
  }
}
// get all possible combinations of a multinomial distribution
template <typename Dtype>
vector<shared_ptr<Blob<Dtype> > > get_combi(int blob_channels, int blob_height, int blob_width, int pooling_size) {
  vector<int> blob_shape(4);
  blob_shape[0] = 1;
  blob_shape[1] = blob_channels;
  blob_shape[2] = blob_height;
  blob_shape[3] = blob_width;
  Blob<Dtype> tmp_blob(blob_shape);
  const int num_combinations = pow(1+pow(pooling_size, 2), tmp_blob.count() / pow(pooling_size, 2));
  vector<int> state((blob_channels * blob_height * blob_width) / pow(pooling_size, 2));
  vector<shared_ptr<Blob<Dtype> > > answer;
  for (int i = 0; i < num_combinations; ++i) {
    answer.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(blob_shape)));
    caffe_set(answer[i]->count(), Dtype(0.), answer[i]->mutable_cpu_data());
    for (int j = 0; j < state.size(); ++j) {
      int offset = pooling_size * blob_width * (j / (blob_width / pooling_size)) + pooling_size * (j % (blob_width / pooling_size));
      if (state[j] > 0) {
        int s = state[j] - 1;
        offset += blob_width * (s / pooling_size);
        offset += s % pooling_size;
        answer[i]->mutable_cpu_data()[offset] = 1;
      }
    }
    update_state(state, pow(pooling_size, 2));
  }
  return answer;
}

template <typename Dtype>
Dtype calculate_energy(const shared_ptr<Blob<Dtype> >& visable_vector,
                       const shared_ptr<Blob<Dtype> >& hidden_vector,
                       RBMCuDNNConvolutionLayer<Dtype>& layer) {
  // Do a forward pass
  Blob<Dtype> output_blob;
  vector<Blob<Dtype>* > output_vector;
  output_vector.push_back(&output_blob);
  vector<Blob<Dtype>* > input_vector;
  input_vector.push_back(visable_vector.get());
  bool starting_fiu = layer.get_forward_is_update();
  layer.set_forward_is_update(false);
  layer.Forward(input_vector, output_vector);
  layer.set_forward_is_update(starting_fiu);
  // multiply matrix product result with hidden vector (which has the fancy pooling going on)
  Dtype matrix_sum = caffe_cpu_dot(hidden_vector->count(),
                      output_vector[0]->cpu_data(), hidden_vector->cpu_data());
  
  // for each filter / output we get one hidden matrix, here we add the filter-specific bias to that and sum
  for (int i = 0; i < hidden_vector->shape()[1]; ++i) {
    matrix_sum += layer.blobs()[1]->cpu_data()[i] * std::accumulate(hidden_vector->cpu_data() + hidden_vector->count(2) * i, hidden_vector->cpu_data() + hidden_vector->count(2) * (i+1), Dtype(0.));
  }

  // Visable bias (which is just a scalar) multiplied with the sum over visable input
  matrix_sum += layer.blobs()[2]->cpu_data()[0] * std::accumulate(visable_vector->cpu_data(), visable_vector->cpu_data() + visable_vector->count(), Dtype(0.));

  // no negative due to double negation
  return exp(matrix_sum);
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

template <typename Dtype>
Dtype calculate_overlap(const vector<shared_ptr<Blob<Dtype> > >& all_visable,
                        const vector<shared_ptr<Blob<Dtype> > >& all_hidden,
                        const vector<Dtype>& actual_probability,
                        RBMCuDNNConvolutionLayer<Dtype>& layer) {
  // now calculate the estimated probability of everything
  vector<Dtype> est_probability(all_visable.size());
  
  Dtype total = 0;
  for (int i = 0; i < all_visable.size(); i++) {
    est_probability[i] = 0;
    for (int j = 0; j < all_hidden.size(); j++) {
      est_probability[i] += calculate_energy(all_visable[i], all_hidden[j], layer);
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

template <typename Dtype>
vector<Dtype> sample_frequency(const vector<shared_ptr<Blob<Dtype> > >& visable,
                               const vector<shared_ptr<Blob<Dtype> > >& hidden,
                               RBMCuDNNConvolutionLayer<Dtype>& layer) {
  Dtype total = 0;
  vector<Dtype> probability(visable.size());

  for (int i = 0; i < visable.size(); i++) {
    probability[i] = 0;
    for (int j = 0; j < hidden.size(); j++) {
      probability[i] += calculate_energy(visable[i], hidden[j], layer);
    }
    total += probability[i];
  }
  for (int i = 0; i < visable.size(); i++) {
    probability[i] /= total;
  }
  return probability;
}

template <typename TypeParam>
class RBMCuDNNConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RBMCuDNNConvolutionLayerTest()
      : blob_bottom_input_(new Blob<Dtype>(10, 3, 4, 9)),
        blob_top_input_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_top_vec_.push_back(blob_top_input_);
  }
  virtual ~RBMCuDNNConvolutionLayerTest() {
    delete blob_bottom_input_;
    delete blob_top_input_;
  }
  
  void generate_layer(int num_output, int num_sample_steps) {
    LayerParameter layer_param;
    RBMConvolutionParameter* rbm_conv_param =
        layer_param.mutable_rbm_convolution_param();
    rbm_conv_param->set_pooling_size(2);
    ConvolutionParameter* conv_param =
        layer_param.mutable_convolution_param();
    conv_param->set_num_output(num_output);
    conv_param->set_kernel_h(2);
    conv_param->set_kernel_w(3);
    conv_param->set_stride_h(2);
    conv_param->set_stride_w(2);
    RBMInnerProductParameter* rbm_inner_product_param =
        layer_param.mutable_rbm_inner_product_param();
    rbm_inner_product_param->set_visable_bias_term(true);
    rbm_inner_product_param->set_sample_steps_in_update(num_sample_steps);
    layer_.reset(new RBMCuDNNConvolutionLayer<Dtype>(layer_param));
  }

  // layer with smaller input and output that we can brute force sample
  void generate_sample_layer(int num_output, int num_sample_steps) {
    LayerParameter layer_param;
    RBMConvolutionParameter* rbm_conv_param =
        layer_param.mutable_rbm_convolution_param();
    rbm_conv_param->set_pooling_size(2);
    ConvolutionParameter* conv_param =
        layer_param.mutable_convolution_param();
    conv_param->set_num_output(num_output);
    conv_param->set_kernel_h(2);
    conv_param->set_kernel_w(2);
    conv_param->set_stride_h(1);
    conv_param->set_stride_w(1);
    RBMInnerProductParameter* rbm_inner_product_param =
        layer_param.mutable_rbm_inner_product_param();
    rbm_inner_product_param->set_visable_bias_term(true);
    rbm_inner_product_param->set_sample_steps_in_update(num_sample_steps);
    layer_.reset(new RBMCuDNNConvolutionLayer<Dtype>(layer_param));
    target_layer_.reset(new RBMCuDNNConvolutionLayer<Dtype>(layer_param));
    
    vector<int> input_shape(4,3);
    input_shape[0] = 10;
    input_shape[1] = 1;
    blob_bottom_input_->Reshape(input_shape);
    layer_->SetUp(blob_bottom_vec_, blob_top_vec_);
    target_layer_->SetUp(blob_bottom_vec_, blob_top_vec_);
    
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    for (int i = 0; i < 3; ++i) {
      filler.Fill(layer_->blobs()[i].get());
      filler.Fill(target_layer_->blobs()[i].get());
    }
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const blob_top_input_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<RBMCuDNNConvolutionLayer<Dtype> > layer_;
  shared_ptr<RBMCuDNNConvolutionLayer<Dtype> > target_layer_;
};

TYPED_TEST_CASE(RBMCuDNNConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestSetUp) {
  this->generate_layer(2,10);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 10);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 9);
  EXPECT_EQ(this->blob_top_input_->num(), 10);
  EXPECT_EQ(this->blob_top_input_->channels(), 2);
  EXPECT_EQ(this->blob_top_input_->height(), 2);
  EXPECT_EQ(this->blob_top_input_->width(), 4);
  ASSERT_EQ(this->layer_->blobs()[0]->shape().size(), 4);
  EXPECT_EQ(this->layer_->blobs()[0]->shape()[0], 2);  // Num output
  ASSERT_EQ(this->layer_->blobs()[1]->shape().size(), 1);
  EXPECT_EQ(this->layer_->blobs()[1]->shape()[0], 2);  // Num output
  ASSERT_EQ(this->layer_->blobs()[2]->shape().size(), 1);
  EXPECT_EQ(this->layer_->blobs()[2]->shape()[0], 1);  // single value
}

TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestForwardSample) {
  typedef typename TypeParam::Dtype Dtype;
  this->generate_layer(2,10);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->SampleForward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data = this->blob_top_input_->cpu_data();
  for (int i = 0; i < this->blob_top_input_->count(); ++i) {
    EXPECT_TRUE(top_data[i] == 0 || top_data[i] == 1);
  }
}

TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestBackwardSample) {
  typedef typename TypeParam::Dtype Dtype;
  this->generate_layer(2,10);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->SampleForward(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->SampleBackward(this->blob_top_vec_, this->blob_bottom_vec_);
  const Dtype* bottom_data = this->blob_bottom_input_->cpu_data();
  for (int i = 0; i < this->blob_bottom_input_->count(); ++i) {
    EXPECT_TRUE(bottom_data[i] == 0 || bottom_data[i] == 1);
  }
}

TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestSample) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1701);
  this->generate_sample_layer(2,10);
  const int num_samples(10000), sample_length(10), batch_size(10);
  
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(1,3,3,1);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(2,2,2,2);
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

  // generate samples from the layer
  for (int i = 0; i < num_samples / batch_size; ++i) {
    // this->blob_bottom_vec_[0] = all_visable[multi[i]].get();
    caffe_set(this->blob_bottom_vec_[0]->count(), Dtype(0.),
              this->blob_bottom_vec_[0]->mutable_cpu_data());
    for (int j = 0; j < sample_length; ++j) {
      this->target_layer_->SampleForward(this->blob_bottom_vec_, this->blob_top_vec_);
      this->target_layer_->SampleBackward(this->blob_top_vec_, this->blob_bottom_vec_);
    }

    // see how often we get different samples
    for (int k = 0; k < batch_size; ++k) {
      ostringstream stuff;
      for (int j = 0; j < this->blob_bottom_input_->count(1); ++j) {
        stuff << "_" << this->blob_bottom_input_->cpu_data()[k*9 + j];
      }
      probability[vector_to_index[stuff.str()]] += Dtype(1.0) / num_samples;
    }
  }

  // make sure the divergence between exact and estimated is low
  Dtype overlap = calculate_overlap(all_visable, all_hidden, probability, *this->target_layer_);
  EXPECT_GE(overlap, .9);
}

// If the weight starts off in the right spot, the update is zero on average
TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestZeroUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1701);
  this->generate_sample_layer(2,20);
  const int num_samples(1000), batch_size(10);

  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(1,3,3,1);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(2,2,2,2);
  const int num_input = all_visable[0]->count();

  vector<Dtype> probability = sample_frequency(all_visable, all_hidden, *this->layer_);

  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  // set the weight diffs to zero
  for (int j = 0; j < 3; ++j) {
    shared_ptr<Blob<Dtype> > blob = this->layer_->blobs()[j];
    caffe_set(blob->count(), Dtype(0.), blob->mutable_cpu_diff());
  }

  vector<int> multi = multinomial(num_samples, probability);
  this->layer_->set_forward_is_update(true);
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
TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestLocalMinimum) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_random_seed(1701);
  this->generate_sample_layer(2,20);
  const int num_samples(1000), batch_size(10);

  // now calculate the probability of each possible visable vector
  vector<shared_ptr<Blob<Dtype> > > all_visable = get_combi<Dtype>(1,3,3,1);
  vector<shared_ptr<Blob<Dtype> > > all_hidden  = get_combi<Dtype>(2,2,2,2);
  const int num_input = all_visable[0]->count();
  Dtype learning_rate = .02 * batch_size;
  vector<Dtype> probability = sample_frequency(all_visable, all_hidden, *this->target_layer_);

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

  vector<Dtype> ground_probability = sample_frequency(all_visable, all_hidden, *this->layer_);

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
        vector<Dtype> new_probability = sample_frequency(all_visable, all_hidden, *this->layer_);

      Dtype new_overlap = 0;
      for (int j = 0; j < probability.size(); ++j) {
        new_overlap += std::min(probability[j], new_probability[j]);
      }
      EXPECT_GE(learned_overlap, new_overlap - .0001 * epsi);
      blob_to_test.mutable_cpu_data()[i] -= epsi;
    }
  }
}

/*
TYPED_TEST(RBMCuDNNConvolutionLayerTest, TestState) {
  typedef typename TypeParam::Dtype Dtype;

  vector<int> state(3);
  for(int i = 0; i < 30; ++i) {
    for(int j = 0; j < state.size(); ++j)
      std::cout << state[j] << " ";
    std::cout << std::endl;
    update_state(state,2);
  }


  int channels(2), height(2), width(4);
  vector<shared_ptr<Blob<Dtype> > > a = get_combi<Dtype>(channels, height, width, 2);
  
  for (int i = 0; i < a.size(); ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          std::cout << a[i]->cpu_data()[c*(height*width)+h*width+w] << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
}*/

}  //  namespace caffe
