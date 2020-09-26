/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/opencl/batch_norm_image_compute.h"

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h" z
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

bool BatchNormCompute::PrepareForRun() {
  bn_param_ = param_.get_mutable<param_t>();
  auto& context = ctx_->As<OpenCLContext>();
  context.cl_context()->AddKernel(kernel_func_name_,
                                  "image/batch_norm_kernel.cl",
                                  build_options_,
                                  time_stamp_);
  STL::stringstream kernel_key;
  kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
  kernel_ = context.cl_context()->GetKernel(kernel_key.str());

  auto input_dims = bn_param_->dims();
  
  new_scale.Resize({input_dims[1]});
  new_bias.Resize({input_dims[1]});

}

void ReInitWhenNeeded() override {
  bn_param_ = param_.get_mutable<param_t>();
  auto x_dims = bn_param_->x->dims();
  if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = x_dims;
    first_epoch_for_reinit_ = false;

    // compute image shape
    paddle::lite::CLImageConverterDefault default_convertor;
    input_img_shape_ =
        default_convertor.InitImageDimInfoWith(bn_param_->X->dims());  // w, h
    out_img_shape_ =
        default_convertor.InitImageDimInfoWith(bn_param_->Out->dims());  // w, h

    // compute global work size
    GetGlobalWorkSize();
  }
}

void GetGlobalWorkSize() {
  global_work_size_ =
      cl::NDRange{static_cast<cl::size_type>(input_img_shape_[0]),
                  static_cast<cl::size_type>(input_img_shape_[1])};
}

void BatchNormCompute::Run() {
  auto kernel = kernel_;
  int index = 0;

  auto* input = bn_param_->x->data<half_t, cl::Image2D>();
  auto* out = bn_param_->
              // auto* new_scale=bn_param->

    cl_int status;
    status = kernel.setArg(index, out_width);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(index++, * input);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(index++, * new_scale);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(index++, * new_bias);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(index++, * out);
    CL_CHECK_FATAL(status);

  auto& context = ctx_->As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);

  status = EnqueueNDRangeKernel(context,
                                kernel,
                                cl::NullRange,
                                global_work_size_,
                                cl::NullRange,
                                nullptr,
                                event_);

  CL_CHECK_FATAL(status);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::BatchNormCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Scale",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Bias",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Mean",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Variance",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("MeanOut",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("VarianceOut",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("SavedMean",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("SavedVariance",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .Finalize();