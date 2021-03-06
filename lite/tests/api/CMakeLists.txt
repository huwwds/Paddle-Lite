if(LITE_WITH_ARM)
    lite_cc_test(test_transformer_with_mask_fp32_arm SRCS test_transformer_with_mask_fp32_arm.cc
        DEPS ${lite_model_test_DEPS} paddle_api_full
        ARM_DEPS ${arm_kernels}
        ARGS --model_dir=${LITE_MODEL_DIR}/transformer_with_mask_fp32 SERIAL)
    if(WITH_TESTING)
        add_dependencies(test_transformer_with_mask_fp32_arm extern_lite_download_transformer_with_mask_fp32_tar_gz)
    endif()
endif()

function(xpu_x86_without_xtcl_test TARGET MODEL DATA)
    if(${DATA} STREQUAL "")
        lite_cc_test(${TARGET} SRCS ${TARGET}.cc
            DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
            ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
            ARGS --model_dir=${LITE_MODEL_DIR}/${MODEL})
    else()
        lite_cc_test(${TARGET} SRCS ${TARGET}.cc
            DEPS mir_passes lite_api_test_helper paddle_api_full paddle_api_light gflags utils
            ${ops} ${host_kernels} ${x86_kernels} ${xpu_kernels}
            ARGS --model_dir=${LITE_MODEL_DIR}/${MODEL} --data_dir=${LITE_MODEL_DIR}/${DATA})
    endif()

    if(WITH_TESTING)
        add_dependencies(${TARGET} extern_lite_download_${MODEL}_tar_gz)
        if(NOT ${DATA} STREQUAL "")
            add_dependencies(${TARGET} extern_lite_download_${DATA}_tar_gz)
        endif()
    endif()
endfunction()

if(LITE_WITH_XPU AND NOT LITE_WITH_XTCL)
    xpu_x86_without_xtcl_test(test_resnet50_fp32_xpu resnet50 ILSVRC2012_small)
    xpu_x86_without_xtcl_test(test_googlenet_fp32_xpu GoogLeNet ILSVRC2012_small)
    xpu_x86_without_xtcl_test(test_vgg19_fp32_xpu VGG19 ILSVRC2012_small)
    xpu_x86_without_xtcl_test(test_ernie_fp32_xpu ernie bert_data)
    xpu_x86_without_xtcl_test(test_bert_fp32_xpu bert bert_data)
endif()

if(LITE_WITH_RKNPU)
    lite_cc_test(test_mobilenetv1_int8_rknpu SRCS test_mobilenetv1_int8_rknpu.cc
        DEPS ${lite_model_test_DEPS} paddle_api_full
        RKNPU_DEPS ${rknpu_kernels} ${rknpu_bridges}
        ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()

if(LITE_WITH_APU)
    lite_cc_test(test_mobilenetv1_int8_apu SRCS test_mobilenetv1_int8_apu.cc
        DEPS ${lite_model_test_DEPS} paddle_api_full
        APU_DEPS ${apu_kernels} ${apu_bridges}
        ARGS --model_dir=${LITE_MODEL_DIR}/MobilenetV1_full_quant SERIAL)
endif()
