// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class BatchNormCompute : public KernelLite<TARGET(kOpenCL),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::BatchNormParam;

  std::string doc() const override {
    return "Batch norm using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void ReInitWhenNeeded() override;
  void PrepareForRun() override;

  void Run() override;

  virtual ~BatchNormCompute() = default;

 private:
  // lite::Tensor new_scale;
  // lite::Tensor new_bias;

  param_t* bn_param_{nullptr};
  DDim input_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim new_scale_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim new_bias_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  DDim last_x_dims_;

  int out_width_;

  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE_half"};
  float threshold_{6.f};
  float scale_{1.f};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};

  Tensor new_scale;
  Tensor new_bias;

  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
