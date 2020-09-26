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
#include <cl_common.h>

__kernel void batchnorm(__private const int out_width,
                        __read_only image2d_t input,
                        __read_only image2d_t new_scale_image,
                        __read_only image2d_t new_bias_image,
                        __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  CL_DTYPE4 new_scale = READ_IMG_TYPE(CL_DTYPE_CHAR, new_scale_image, sampler, (int2)(out_c, 0));
  CL_DTYPE4 new_bias = READ_IMG_TYPE(CL_DTYPE_CHAR, new_bias_image, sampler, (int2)(out_c, 0));

  int pos_x = mad24(out_c, out_width, out_w);
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_x, out_nh));
  CL_DTYPE4 out = mad(in, new_scale, new_bias);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_x, out_nh), out);
}