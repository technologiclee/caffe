#ifndef CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_

#include "caffe/layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Fully connected RBM layer, with both hidden and visible biases
 *
 * The layer takes up to two bottom blobs.  The first blob just specifies the
 * input data during forward pass, update and forward sample.  During backward
 * sampling, the first bottom blob's diff gets the backward propigated
 * probabilites, and the blob's data gets samples from these probabilites.
 *
 * If a second bottom blob is set, it is the "clamp." The clamp is a blob the
 * same size as the first (input data) blob and only has an effect during
 * sampling.  For any index, if the clamp value is set to one, then backward
 * sampling does not change the input data at that index.
 *
 * The top blobs are similar to the bottom blobs.  The frirst two are the
 * hidden units and the hidden clamps.  The only difference is the third blob,
 * if specified is a loss blob, and during the update phase the reconstruction
 * of the image is saved there
 *
 * The layer also contains two bias terms.  The hidden bias term is the same as
 * the bias term in the InnerProductLayer, and the visible bias term is new.
 *
 */
template <typename Dtype>
class RBMInnerProductLayer : public Layer<Dtype> {
 public:
  explicit RBMInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBMInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int num_visible_;
  int num_hidden_;
  int batch_size_;

  int num_sample_steps_for_update_;
  bool visible_bias_term_;

  int visible_bias_index_;
  Blob<Dtype> bias_multiplier_;

 private:
  void sample_h_given_v(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
  void sample_v_given_h(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
  void gibbs_hvh(const vector<Blob<Dtype>*>& bottom,
                 const vector<Blob<Dtype>*>& top);
  void update_diffs(const int k, const vector<Blob<Dtype>*>& hidden_k,
                    const vector<Blob<Dtype>*>& visible_k);
  // The layer which is used to fist process the input.
  shared_ptr<Layer<Dtype> > connection_layer_;
  vector<Blob<Dtype>*> pre_activation_h1_vec_;
  vector<Blob<Dtype>*> pre_activation_v1_vec_;
  // The layer is used as an activation after the input has been passed through
  // the connection layer.
  shared_ptr<Layer<Dtype> > activation_layer_;
  vector<Blob<Dtype>*> mean_h1_vec_;
  vector<Blob<Dtype>*> mean_v1_vec_;
  // A layer used to sample the activations.
  shared_ptr<Layer<Dtype> > sampling_layer_;
  vector<Blob<Dtype>*> sample_h1_vec_;
  vector<Blob<Dtype>*> sample_v1_vec_;

  // TODO: This is part of the sampling layer.
  shared_ptr<SyncedMemory> rng_data_;
};
}

#endif  // CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_