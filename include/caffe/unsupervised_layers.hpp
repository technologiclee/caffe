#ifndef CAFFE_UNSUPERVISED_LAYERS_HPP_
#define CAFFE_UNSUPERVISED_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Interface for layers that can be learned in an unsupervised setting
 *
 * UnsupervisedLayer%s must implement an Update function, in which they take
 * input from their bottom Blob%s and use this input to create a better
 * internal representation of the data.  The update may then optionally pass
 * data to the next layer through its top
 */
template <typename Dtype>
class UnsupervisedLayer : public virtual Layer<Dtype> {
 public:
  explicit UnsupervisedLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~UnsupervisedLayer() {}

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     optional outputs. If the length of the top vector is the same as
   *     MaxTopBlobs, then the last blob will always be some error blob.
   * \return The total loss from the layer.
   *
   * The Update wrapper calls the relevant device wrapper function
   * (Update_cpu or Update_gpu) to update the layer's representation of the
   * data.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Update_cpu and (optionally) Update_gpu.
   */
  virtual Dtype Update(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  /** @brief Using the CPU device, do an update of the unsupervised model. */
  virtual void Update_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, do an update of the unsupervised model.
   *        Fall back to Update_cpu() if unavailable.
   */
  virtual void Update_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    return Update_cpu(bottom, top);
  }
};

/**
 * @brief Interface for layers that can be used to generate samples of the data
 *
 * GenerativeUnsupervisedLayer%s must implement the SampleForward and
 * SampleBackward functions, which given either top or bottom inputs creates a
 * sample for the other.  The sample functions can optionally also do a forward
 * or backward pass of the values without sampling, if one wants both the
 * sampled and unsampled data
 */
template <typename Dtype>
class GenerativeUnsupervisedLayer : public UnsupervisedLayer<Dtype> {
 public:
  explicit GenerativeUnsupervisedLayer(const LayerParameter& param)
      : Layer<Dtype>(param), UnsupervisedLayer<Dtype>(param) {}
  virtual ~GenerativeUnsupervisedLayer() {}

  /**
   * @brief Given the bottom blobs, compute a sample of the top blobs.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     sample and perhaps forward pass outputs. If a second blob with 0-1
   *     values of the same dimensions is set, this blob acts as a clamp and
   *     where ever the clamp is one, the sampling does not effect the value
   *     of the hidden data
   *
   * The Sample wrapper calls the relevant device wrapper function
   * (SampleForward_cpu or Update_gpu) to update the layer's representation of the
   * data.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement SampleForward_cpu and (optionally)
   * SampleForward_gpu.
   */
  virtual void SampleForward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blobs, compute a sample of the bottom blobs.
   * @see GenerativeUnsupervisedLayer::SampleForward
   */
  virtual void SampleBackward(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom);

 protected:
  /** @brief Using the CPU device, forward sample the unsupervised model. */
  virtual void SampleForward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /** @brief Using the CPU device, backward sample the unsupervised model. */
  virtual void SampleBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) = 0;

  /**
   * @brief Using the GPU device, forward sample the unsupervised model.
   *        Fall back to SampleForward_cpu() if unavailable.
   */
  virtual void SampleForward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    SampleForward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, backward sample the unsupervised model.
   *        Fall back to SampleBackward_cpu() if unavailable.
   */
  virtual void SampleBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<Blob<Dtype>*>& bottom) {
    SampleBackward_cpu(top, bottom);
  }
};

/**
 * @brief Fully connected RBM layer, with both hidden and visable biases
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
 * the bias term in the InnerProductLayer, and the visable bias term is new.
 *
 */
template <typename Dtype>
class RBMInnerProductLayer :
public GenerativeUnsupervisedLayer<Dtype>, public InnerProductLayer<Dtype> {
 public:
  explicit RBMInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        GenerativeUnsupervisedLayer<Dtype>(param),
        InnerProductLayer<Dtype>(param) {}
  virtual ~RBMInnerProductLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBMInnerProduct"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  /**
   * @brief Use contrastive divergence to find an update step for the weights
   * @param bottom Input data for the visable layer that will be used for the
   *        update. Only the first blob is used.
   * @param top Hidden output of the update.  The first blob is just the result
   *        of the forward pass, and the second blob has no effect and the last
   *        if specified contains the absolute reconstruction error of the data
   */
  virtual void Update_cpu(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Update_gpu(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void SampleForward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

  virtual void SampleBackward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<Blob<Dtype>*>& bottom);

  virtual void SampleForward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

  virtual void SampleBackward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  bool visable_bias_term_;
  int num_sample_steps_for_update_;
  int visable_bias_index_;
 private:
  shared_ptr<SyncedMemory> rng_data_;
};

#ifdef USE_CUDNN
/**
 * @brief Convolutional RBM layer, with both hidden and visable biases
 * 
 * TODO: this should be split into RBMConvolutionLayer and RBMCuDNNConvolutionLayer
 * and you should decide between the two by setting ENGINE. See how the layer factory
 * does this for the ConvolutionLayer
 * 
 * Ok, so this is pretty similar to the RBM layer, except for the fact that we use all
 * the convolutional stuff.  There are some key differences though:
 * 
 * 1) for now we can't clamp the hidden outputs.  This is just to make the implementation
 *    easier
 * 
 * 2) the sampling of hidden units is different, with this stochastic sampling described in
 *    the paper
 * 
 * 3) the hidden biases are a different shape, and we only have one bias per output filter,
 *    not one bias per output value as before
 * 
 * 4) There are no visable biases.  Make sure that your input starts off centered
 * 
 * TODO:
 *   - I divide by batch size in the update a lot in the normal rbm case.  Is this correct?
 *      -> It is def an error, I need to change this in the normal RBM case
 *   - Perhaps change visable bias to a 4d tensor, use CUDNN to update and the like
 *   - I'm reshaping the bias_shape by N_ in the normal RBM case which is wrong!!!
 *   - What shape should the visable bias have?
 *   - What shape should be hidden bias have?  -- I think the current shape is wrong
 *   - is K_ correctly used everywhere?
 *   - Why is the bias done the way it is hin conv layer?
 *   - In the latest version of caffe the bias has a different form
 *     -> also the paper speaks of one bias per filter
 *   - I think the biases for the visable vectors should also have some sort of a convolutional element
 */
template <typename Dtype>
class RBMCuDNNConvolutionLayer :
public GenerativeUnsupervisedLayer<Dtype>, public CuDNNConvolutionLayer<Dtype> {
 public:
  explicit RBMCuDNNConvolutionLayer(const LayerParameter& param)
      : GenerativeUnsupervisedLayer<Dtype>(param),
        CuDNNConvolutionLayer<Dtype>(param),
        Layer<Dtype>(param) {}
  virtual ~RBMCuDNNConvolutionLayer() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBMCuDNNConvolution"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }
  inline bool get_forward_is_update() const {return forward_is_update_;}
  inline void set_forward_is_update(bool val){forward_is_update_ = val;}
protected:
  /**
   * @brief Use contrastive divergence to find an update step for the weights
   * @param bottom Input data for the visable layer that will be used for the
   *        update. Only the first blob is used.
   * @param top Hidden output of the update.  The first blob is just the result
   *        of the forward pass, and the second blob has no effect and the last
   *        if specified contains the absolute reconstruction error of the data
   */
  virtual void Update_cpu(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  
  virtual void Update_gpu(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void SampleForward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

  virtual void SampleBackward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<Blob<Dtype>*>& bottom);

  virtual void SampleForward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

  virtual void SampleBackward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<Blob<Dtype>*>& bottom);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  
  int num_sample_steps_for_update_;
  int pooling_size_;
  int visable_bias_index_;
  int num_errors_;
  bool forward_is_update_;
 private:
  shared_ptr<SyncedMemory> rng_data_;
};
#endif

// Update wrapper. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype UnsupervisedLayer<Dtype>::Update(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  this->Reshape(bottom, top);

  switch (Caffe::mode()) {
  case Caffe::CPU:
    this->Update_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    this->Update_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return loss;
}

// SampleBackward wrapper. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void GenerativeUnsupervisedLayer<Dtype>::SampleBackward(
    const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) {
  // this->Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    SampleBackward_cpu(top, bottom);
    break;
  case Caffe::GPU:
    SampleBackward_gpu(top, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// SampleForward wrapper. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline void GenerativeUnsupervisedLayer<Dtype>::SampleForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    SampleForward_cpu(bottom, top);
    break;
  case Caffe::GPU:
    SampleForward_gpu(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

}  // namespace caffe

#endif  // CAFFE_UNSUPERVISED_LAYERS_HPP_
