#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"

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
        blob_top_input_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_input_);
    blob_bottom_vec_.push_back(blob_bottom_input_);
    blob_top_vec_.push_back(blob_top_input_);
  }
  virtual ~RBMInnerProductLayerTest() {
    delete blob_bottom_input_;
    delete blob_top_input_;
  }

  void fill_gaussian(Blob<Dtype>* fill_me) {
    FillerParameter filler_param;
    filler_param.set_type("gaussian");
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(fill_me);
  }

  Blob<Dtype>* const blob_bottom_input_;
  Blob<Dtype>* const blob_top_input_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RBMInnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(RBMInnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_inner_product_param()->set_num_output(10);

  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<int> expected_in(4);
  expected_in[0] = 2; expected_in[1] = 3; expected_in[2] = 4; expected_in[3] = 5;
  EXPECT_EQ(this->blob_bottom_input_->shape(), expected_in);

  vector<int> expected_out(2);
  expected_out[0] = 2; expected_out[1] = 10;
  EXPECT_EQ(this->blob_top_input_->shape(), expected_out);
}


TYPED_TEST(RBMInnerProductLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_inner_product_param()->set_num_output(10);

  shared_ptr<RBMInnerProductLayer<Dtype> > layer(
      new RBMInnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  vector<int> expected_in(4);
  expected_in[0] = 6; expected_in[1] = 3; expected_in[2] = 4; expected_in[3] = 5;

  this->blob_bottom_input_->Reshape(expected_in);
  layer->Reshape(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_bottom_input_->shape(), expected_in);

  vector<int> expected_out(2);
  expected_out[0] = 6; expected_out[1] = 10;
  EXPECT_EQ(this->blob_top_input_->shape(), expected_out);
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

  const Dtype* test_data = this->blob_top_vec_[0]->cpu_data();
  const int N = this->blob_top_vec_[0]->count();
  for (int i = 0; i < N; ++i) {
    EXPECT_GE(test_data[i], 0);
    EXPECT_LE(test_data[i], 1);
  }
}

}  //  namespace caffe
