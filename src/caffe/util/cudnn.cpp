#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"

namespace caffe {
namespace cudnn {

float dataType<float>::oneval = 1.0;
float dataType<float>::zeroval = 0.0;
float dataType<float>::minusoneval = -1.0;
const void* dataType<float>::one =
    static_cast<void *>(&dataType<float>::oneval);
const void* dataType<float>::zero =
    static_cast<void *>(&dataType<float>::zeroval);
const void* dataType<float>::minusone =
    static_cast<void *>(&dataType<float>::minusoneval);

double dataType<double>::oneval = 1.0;
double dataType<double>::zeroval = 0.0;
double dataType<double>::minusoneval = -1.0;
const void* dataType<double>::one =
    static_cast<void *>(&dataType<double>::oneval);
const void* dataType<double>::zero =
    static_cast<void *>(&dataType<double>::zeroval);
const void* dataType<double>::minusone =
    static_cast<void *>(&dataType<double>::minusoneval);

}  // namespace cudnn
}  // namespace caffe
#endif
