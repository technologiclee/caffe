#ifndef CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
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
 * The top contains then one to three blobs describing the hidden state of the
 * RBM followed by an arbitrary number of blobs with different error values.
 *
 * The layer also contains two bias terms.  The hidden bias term is the same as
 * the bias term in the InnerProductLayer, and the visible bias term is new.
 * The backwards pass of this RBM is just the forward pass of the 
 * InnerProductLayer with the weight matrix transposed and the bias being the
 * visible bias
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
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  /**
   * Performing a forward pass either performs an unsupervised update if
   * forward_is_update is set to true in the prototxt (or set to true by the
   * setter or performs a normal forward pass if it is set to false.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
