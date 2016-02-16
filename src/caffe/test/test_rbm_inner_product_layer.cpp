#include <algorithm>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/util/rng.hpp"

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
        pre_activation_h1_(new Blob<Dtype>()),
        post_activation_h1_(new Blob<Dtype>()),
        sample_h1_(new Blob<Dtype>()),
        blob_top_error_1_(new Blob<Dtype>()),
        blob_top_error_2_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
  }

  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete pre_activation_h1_;
    delete post_activation_h1_;
    delete sample_h1_;
    delete blob_top_error_1_;
    delete blob_top_error_2_;
  }

  virtual void InitLayerFromProtoString(const string& proto) {
    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    layer_.reset(new RBMInnerProductLayer<Dtype>(layer_param));
  }

  virtual string getLayerText(const string& extra_text = "",
                              bool forward_is_update = true) {
    string proto =
      "name: 'rbm_inner_product_layer' "
      "type: 'RBMInnerProduct' "
      "rbm_inner_product_param { "
      "  sample_steps_in_update: 2 "
      "  connection_layer_param { "
      "    name: 'connection_inner_product' "
      "    type: 'InnerProduct' "
      "    inner_product_param { "
      "      num_output: 10 "
      "      bias_term: true "
      "    } "
      "  } "
      "  hidden_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  hidden_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } "
      "  visible_bias_term: true ";
    if (forward_is_update) {
      proto += "  forward_is_update: true ";
    } else {
      proto += "  forward_is_update: false ";
    }
    proto += extra_text;
    proto += "} ";
    return proto;
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const pre_activation_h1_;
  Blob<Dtype>* const post_activation_h1_;
  Blob<Dtype>* const sample_h1_;
  Blob<Dtype>* const blob_top_error_1_;
  Blob<Dtype>* const blob_top_error_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<RBMInnerProductLayer<Dtype> > layer_;
};

TYPED_TEST_CASE(RBMInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpNoVisibleActivation) {
  string proto = this->getLayerText();
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  string extra_text =
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

TYPED_TEST(RBMInnerProductLayerTest, TestSetUpWithVisibleActivation) {
    string extra_text =
      "  visible_activation_layer_param { "
      "    name: 'hidden_activation' "
      "    type: 'Sigmoid' "
      "  } "
      "  visible_sampling_layer_param { "
      "    name: 'hidden_sample' "
      "    type: 'BernoulliSample' "
      "  } ";

  string proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);

  this->blob_top_vec_.clear();
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->num(), 2);
  EXPECT_EQ(this->blob_bottom_input_->channels(), 3);
  EXPECT_EQ(this->blob_bottom_input_->height(), 4);
  EXPECT_EQ(this->blob_bottom_input_->width(), 5);

  // add something to the top
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->pre_activation_h1_->num(), 2);
  EXPECT_EQ(this->pre_activation_h1_->channels(), 10);
  EXPECT_EQ(this->pre_activation_h1_->height(), 1);
  EXPECT_EQ(this->pre_activation_h1_->width(), 1);

  // add more to the top
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->post_activation_h1_->num(), 2);
  EXPECT_EQ(this->post_activation_h1_->channels(), 10);
  EXPECT_EQ(this->post_activation_h1_->height(), 1);
  EXPECT_EQ(this->post_activation_h1_->width(), 1);

  // add even more to the top
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->sample_h1_->num(), 2);
  EXPECT_EQ(this->sample_h1_->channels(), 10);
  EXPECT_EQ(this->sample_h1_->height(), 1);
  EXPECT_EQ(this->sample_h1_->width(), 1);

  // test that error tops are resized correctly
  extra_text +=
      "  loss_measure: RECONSTRUCTION "
      "  loss_measure: FREE_ENERGY ";
  proto = this->getLayerText(extra_text);
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->blob_top_vec_.push_back(this->blob_top_error_2_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_error_1_->num(), 2);
  EXPECT_EQ(this->blob_top_error_1_->channels(), 3);
  EXPECT_EQ(this->blob_top_error_1_->height(), 4);
  EXPECT_EQ(this->blob_top_error_1_->width(), 5);

  EXPECT_EQ(this->blob_top_error_2_->num(), 2);
  EXPECT_EQ(this->blob_top_error_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_error_2_->height(), 1);
  EXPECT_EQ(this->blob_top_error_2_->width(), 1);
}

TYPED_TEST(RBMInnerProductLayerTest, TestForwardNoUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  string extra_text = "  loss_measure: FREE_ENERGY ";
  string proto = this->getLayerText(extra_text, false);

  // run forward with no non error output
  this->InitLayerFromProtoString(proto);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // create an inner product layer that is a copy of this one
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);

  string ip_proto =
      "name: 'inner_product_layer' "
      "type: 'InnerProduct' "
      "inner_product_param { "
      "  num_output: 10 "
      "  bias_term: true "
      "} ";
  LayerParameter layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(ip_proto, &layer_param));
  InnerProductLayer<Dtype> ip_layer(layer_param);
  Blob<Dtype> ip_top_blob;
  vector<Blob<Dtype>*> ip_top_vec;
  ip_top_vec.push_back(&ip_top_blob);
  ip_layer.SetUp(this->blob_bottom_vec_, ip_top_vec);

  // copy the weights so they are the same
  caffe_copy(ip_layer.blobs()[0]->count(), this->layer_->blobs()[0]->cpu_data(),
             ip_layer.blobs()[0]->mutable_cpu_data());
  caffe_copy(ip_layer.blobs()[1]->count(), this->layer_->blobs()[1]->cpu_data(),
             ip_layer.blobs()[1]->mutable_cpu_data());

  // do a forward with both layers
  ip_layer.Forward(this->blob_bottom_vec_, ip_top_vec);

  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  ASSERT_EQ(ip_top_blob.count(), this->pre_activation_h1_->count());
  // make sure the data is the same
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->pre_activation_h1_->cpu_data()[i]);
  }

  // now do a forward and a squash
  SigmoidLayer<Dtype> sigmoid_layer(layer_param);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(ip_top_blob.count(), this->pre_activation_h1_->count());
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->pre_activation_h1_->cpu_data()[i]);
  }
  sigmoid_layer.SetUp(ip_top_vec, ip_top_vec);
  sigmoid_layer.Forward(ip_top_vec, ip_top_vec);
  ASSERT_EQ(ip_top_blob.count(), this->post_activation_h1_->count());
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_FLOAT_EQ(ip_top_blob.cpu_data()[i],
                    this->post_activation_h1_->cpu_data()[i]);
  }

  // check that the sampling really gives us just zeros and ones
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(this->pre_activation_h1_);
  this->blob_top_vec_.push_back(this->post_activation_h1_);
  this->blob_top_vec_.push_back(this->sample_h1_);
  this->blob_top_vec_.push_back(this->blob_top_error_1_);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < ip_top_blob.count(); ++i) {
    EXPECT_TRUE((this->sample_h1_->cpu_data()[i] == 0 ||
                 this->sample_h1_->cpu_data()[i] == 1));
  }
}

}  // namespace caffe
