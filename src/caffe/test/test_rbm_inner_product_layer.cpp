#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/unsupervised_layers.hpp"


#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class RBMInnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RBMInnerProductLayerTest()
      : blob_bottom_input_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_clamp_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_input_(new Blob<Dtype>()),
        blob_top_clamp_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_top_error_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_clamp_);
    blob_top_vec_.push_back(blob_top_input_);
    blob_top_vec_.push_back(blob_top_clamp_);
    blob_top_vec_.push_back(blob_top_error_);
  }
  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete blob_bottom_clamp_;
    delete blob_top_input_;
    delete blob_top_clamp_;
    delete blob_top_error_;
  }

  void fill_gaussian(Blob<Dtype>* fill_me) {
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(fill_me);
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const blob_bottom_clamp_;
  Blob<Dtype>* const blob_top_input_;
  Blob<Dtype>* const blob_top_clamp_;
  Blob<Dtype>* const blob_top_error_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
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
  EXPECT_EQ(this->blob_bottom_clamp_->num(), 2);
  EXPECT_EQ(this->blob_bottom_clamp_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_clamp_->height(), 4);
  EXPECT_EQ(this->blob_bottom_clamp_->width(), 5);
  EXPECT_EQ(this->blob_top_input_->num(), 2);
  EXPECT_EQ(this->blob_top_input_->channels(), 10);
  EXPECT_EQ(this->blob_top_input_->height(), 1);
  EXPECT_EQ(this->blob_top_input_->width(), 1);
  EXPECT_EQ(this->blob_top_clamp_->num(), 2);
  EXPECT_EQ(this->blob_top_clamp_->channels(), 10);
  EXPECT_EQ(this->blob_top_clamp_->height(), 1);
  EXPECT_EQ(this->blob_top_clamp_->width(), 1);
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
  shared_ptr<SigmoidLayer<Dtype> > squash(new SigmoidLayer<Dtype>(LayerParameter()));

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

TYPED_TEST(RBMInnerProductLayerTest, TestBackward) {
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
  vector<bool> false_vector(2, false);
  layer->Backward(this->blob_top_vec_, false_vector, this->blob_bottom_vec_);

  for (int i = 0; i < this->blob_bottom_input_->count(); ++i) {
    EXPECT_GE(this->blob_bottom_input_->cpu_data()[i], 0);
    EXPECT_LE(this->blob_bottom_input_->cpu_data()[i], 1);
    EXPECT_GE(this->blob_bottom_input_->cpu_diff()[i], 0);
    EXPECT_LE(this->blob_bottom_input_->cpu_diff()[i], 1);
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestForwardSampleNoClamp) {
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

TYPED_TEST(RBMInnerProductLayerTest, TestForwardSampleWithClamp) {
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
  Blob<Dtype> clamp(this->blob_top_input_->shape());
  for (int i = 0; i < clamp.count(); ++i) {
    clamp.mutable_cpu_data()[i] = i % 2;
  }
  top.push_back(&clamp);

  this->fill_gaussian(this->blob_top_input_);
  Blob<Dtype> top_copy(this->blob_top_input_->shape());
  const Dtype* top_data = top[0]->cpu_data();
  caffe_copy(top_copy.count(), top_data, top_copy.mutable_cpu_data());

  layer->SampleForward(bottom, top);
  top_data = top[0]->cpu_data();
  for (int i = 0; i < this->blob_top_input_->count(); ++i) {
    if (i % 2 == 0) {
      EXPECT_TRUE(top_data[i] == 0 || top_data[i] == 1);
    } else {
      EXPECT_EQ(top_data[i], top_copy.cpu_data()[i]);
    }
  }
}

TYPED_TEST(RBMInnerProductLayerTest, TestBackwardSampleNoClamp) {
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

TYPED_TEST(RBMInnerProductLayerTest, TestBackwardSampleWithClamp) {
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

  Blob<Dtype> clamp(this->blob_bottom_input_->shape());
  for (int i = 0; i < clamp.count(); ++i) {
    clamp.mutable_cpu_data()[i] = i % 2;
  }
  bottom.push_back(&clamp);
  Blob<Dtype> bot_copy(this->blob_bottom_input_->shape());
  const Dtype* bottom_data = this->blob_bottom_input_->cpu_data();
  caffe_copy(bot_copy.count(), bottom_data, bot_copy.mutable_cpu_data());

  layer->SampleBackward(top, bottom);
  bottom_data = this->blob_bottom_input_->cpu_data();

  for (int i = 0; i < this->blob_bottom_input_->count(); ++i) {
    if (i % 2 == 0) {
      EXPECT_TRUE(bottom_data[i] == 0 || bottom_data[i] == 1);
    } else {
      EXPECT_EQ(bottom_data[i], bot_copy.cpu_data()[i]);
    }
  }
}

}  //  namespace caffe
