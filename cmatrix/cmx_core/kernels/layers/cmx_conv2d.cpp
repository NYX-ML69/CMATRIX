#include "cmx_conv2d.hpp"
#include "cmx_kernel_registry.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>

#ifdef ARM_MATH_CM4
#include "arm_math.h"
#endif

namespace cmx {
namespace kernels {

struct CmxConv2D::Conv2DImpl {
    Conv2DParams params;
    TensorDescriptor input_desc;
    TensorDescriptor weight_desc;
    TensorDescriptor bias_desc;
    TensorDescriptor output_desc;
    
    // Computed values
    int32_t batch_size;
    int32_t input_height;
    int32_t input_width;
    int32_t output_height;
    int32_t output_width;
    
    // Workspace requirements
    int32_t workspace_size;
    
    // Optimized kernel function pointer
    void (*kernel_func)(const float*, const float*, const float*, float*, 
                       const Conv2DParams&, const Conv2DImpl&);
    
    Conv2DImpl() : workspace_size(0), kernel_func(nullptr) {}
};

CmxConv2D::CmxConv2D() : impl_(new Conv2DImpl()) {}

CmxConv2D::~CmxConv2D() {
    delete impl_;
}

KernelStatus CmxConv2D::configure(
    const std::vector<TensorDescriptor>& inputs,
    std::vector<TensorDescriptor>& outputs,
    const void* params
) {
    if (!params) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    const Conv2DParams& conv_params = *static_cast<const Conv2DParams*>(params);
    
    // Validate parameters
    KernelStatus status = validate_params(inputs, conv_params);
    if (status != KernelStatus::SUCCESS) {
        return status;
    }
    
    // Store configuration
    impl_->params = conv_params;
    impl_->input_desc = inputs[0];
    impl_->weight_desc = inputs[1];
    if (conv_params.use_bias && inputs.size() > 2) {
        impl_->bias_desc = inputs[2];
    }
    
    // Extract input dimensions based on data format
    const TensorShape& input_shape = inputs[0].shape;
    if (conv_params.data_format == DataFormat::NHWC) {
        impl_->batch_size = input_shape.dims[0];
        impl_->input_height = input_shape.dims[1];
        impl_->input_width = input_shape.dims[2];
    } else if (conv_params.data_format == DataFormat::NCHW) {
        impl_->batch_size = input_shape.dims[0];
        impl_->input_height = input_shape.dims[2];
        impl_->input_width = input_shape.dims[3];
    } else {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Compute output dimensions
    auto output_size = compute_output_size(
        impl_->input_height, impl_->input_width, conv_params);
    impl_->output_height = output_size.first;
    impl_->output_width = output_size.second;
    
    // Configure output tensor
    if (outputs.empty()) {
        outputs.resize(1);
    }
    
    TensorShape output_shape;
    if (conv_params.data_format == DataFormat::NHWC) {
        output_shape.dims = {impl_->batch_size, impl_->output_height, 
                            impl_->output_width, conv_params.output_channels};
    } else {
        output_shape.dims = {impl_->batch_size, conv_params.output_channels,
                            impl_->output_height, impl_->output_width};
    }
    
    outputs[0].shape = output_shape;
    outputs[0].dtype = inputs[0].dtype;
    impl_->output_desc = outputs[0];
    
    // Compute workspace requirements
    impl_->workspace_size = impl_->output_height * impl_->output_width * 
                           conv_params.input_channels * sizeof(float);
    
    return KernelStatus::SUCCESS;
}

KernelStatus CmxConv2D::run(
    const void* const* inputs,
    void* const* outputs
) {
    if (!inputs || !outputs || !inputs[0] || !outputs[0]) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Dispatch to appropriate implementation based on data format
    if (impl_->params.data_format == DataFormat::NHWC) {
        return run_nhwc(inputs, outputs);
    } else if (impl_->params.data_format == DataFormat::NCHW) {
        return run_nchw(inputs, outputs);
    }
    
    return KernelStatus::INVALID_PARAMS;
}

std::vector<TensorShape> CmxConv2D::infer_shape(
    const std::vector<TensorShape>& input_shapes,
    const void* params
) {
    if (!params || input_shapes.empty()) {
        return {};
    }
    
    const Conv2DParams& conv_params = *static_cast<const Conv2DParams*>(params);
    const TensorShape& input_shape = input_shapes[0];
    
    int32_t batch_size, input_height, input_width;
    
    if (conv_params.data_format == DataFormat::NHWC) {
        batch_size = input_shape.dims[0];
        input_height = input_shape.dims[1];
        input_width = input_shape.dims[2];
    } else {
        batch_size = input_shape.dims[0];
        input_height = input_shape.dims[2];
        input_width = input_shape.dims[3];
    }
    
    auto output_size = compute_output_size(input_height, input_width, conv_params);
    
    TensorShape output_shape;
    if (conv_params.data_format == DataFormat::NHWC) {
        output_shape.dims = {batch_size, output_size.first, output_size.second, 
                            conv_params.output_channels};
    } else {
        output_shape.dims = {batch_size, conv_params.output_channels,
                            output_size.first, output_size.second};
    }
    
    return {output_shape};
}

bool CmxConv2D::supports_dtype(DataType dtype) const {
    return dtype == DataType::FLOAT32 || dtype == DataType::INT8 || 
           dtype == DataType::UINT8;
}

int32_t CmxConv2D::get_workspace_size() const {
    return impl_->workspace_size;
}

KernelStatus CmxConv2D::validate_params(
    const std::vector<TensorDescriptor>& inputs,
    const Conv2DParams& params
) {
    // Check minimum input count
    if (inputs.size() < 2) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Check bias input if required
    if (params.use_bias && inputs.size() < 3) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Validate kernel dimensions
    if (params.kernel_height <= 0 || params.kernel_width <= 0) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Validate stride values
    if (params.stride_height <= 0 || params.stride_width <= 0) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Validate dilation values
    if (params.dilation_height <= 0 || params.dilation_width <= 0) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Validate channel counts
    if (params.input_channels <= 0 || params.output_channels <= 0) {
        return KernelStatus::INVALID_PARAMS;
    }
    
    // Validate input tensor shapes
    const TensorShape& input_shape = inputs[0].shape;
    const TensorShape& weight_shape = inputs[1].shape;
    
    if (input_shape.rank() != 4 || weight_shape.rank() != 4) {
        return KernelStatus::SHAPE_MISMATCH;
    }
    
    // Check weight tensor dimensions
    if (weight_shape.dims[0] != params.output_channels ||
        weight_shape.dims[1] != params.kernel_height ||
        weight_shape.dims[2] != params.kernel_width ||
        weight_shape.dims[3] != params.input_channels) {
        return KernelStatus::SHAPE_MISMATCH;
    }
    
    return KernelStatus::SUCCESS;
}

std::pair<int32_t, int32_t> CmxConv2D::compute_output_size(
    int32_t input_height,
    int32_t input_width,
    const Conv2DParams& params
) {
    int32_t pad_top = params.pad_top;
    int32_t pad_bottom = params.pad_bottom;
    int32_t pad_left = params.pad_left;
    int32_t pad_right = params.pad_right;
    
    // Handle SAME padding mode
    if (params.padding_mode == PaddingMode::SAME) {
        auto h_padding = compute_same_padding(
            input_height, params.kernel_height, params.stride_height, params.dilation_height);
        auto w_padding = compute_same_padding(
            input_width, params.kernel_width, params.stride_width, params.dilation_width);
        
        pad_top = h_padding.first;
        pad_bottom = h_padding.second;
        pad_left = w_padding.first;
        pad_right = w_padding.second;
    }
    
    // Compute effective kernel size with dilation
    int32_t effective_kernel_height = params.kernel_height + 
                                     (params.kernel_height - 1) * (params.dilation_height - 1);
    int32_t effective_kernel_width = params.kernel_width + 
                                    (params.kernel_width - 1) * (params.dilation_width - 1);
    
    // Compute output dimensions
    int32_t padded_height = input_height + pad_top + pad_bottom;
    int32_t padded_width = input_width + pad_left + pad_right;
    
    int32_t output_height = (padded_height - effective_kernel_height) / params.stride_height + 1;
    int32_t output_width = (padded_width - effective_kernel_width) / params.stride_width + 1;
    
    return {output_height, output_width};
}

std::pair<int32_t, int32_t> CmxConv2D::compute_same_padding(
    int32_t input_size,
    int32_t kernel_size,
    int32_t stride,
    int32_t dilation
) {
    int32_t effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1);
    int32_t output_size = (input_size + stride - 1) / stride;
    int32_t total_padding = std::max(0, (output_size - 1) * stride + effective_kernel_size - input_size);
    
    int32_t pad_before = total_padding / 2;
    int32_t pad_after = total_padding - pad_before;
    
    return {pad_before, pad_after};
}

void CmxConv2D::apply_activation(float* data, int32_t size, ActivationType activation) {
    switch (activation) {
        case ActivationType::RELU:
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::max(0.0f, data[i]);
            }
            break;
        case ActivationType::RELU6:
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::min(6.0f, std::max(0.0f, data[i]));
            }
            break;
        case ActivationType::TANH:
            for (int32_t i = 0; i < size; ++i) {
                data[i] = std::tanh(data[i]);
            }
            break;
        case ActivationType::SIGMOID:
            for (int32_t i = 0; i < size; ++i) {
                data[i] = 1.0f / (1.0f + std::exp(-data[i]));
            }
            break;
        case ActivationType::SWISH:
            for (int32_t i = 0; i < size; ++i) {
                data[i] = data[i] / (1.0f + std::exp(-data[i]));
            }
            break;
        case ActivationType::NONE:
        default:
            break;
    }
}

KernelStatus CmxConv2D::run_nhwc(
    const void* const* inputs,
    void* const* outputs
) {
    const float* input = static_cast<const float*>(inputs[0]);
    const float* weights = static_cast<const float*>(inputs[1]);
    const float* bias = impl_->params.use_bias ? static_cast<const float*>(inputs[2]) : nullptr;
    float* output = static_cast<float*>(outputs[0]);
    
    const Conv2DParams& p = impl_->params;
    
    // Compute padding
    int32_t pad_top = p.pad_top;
    int32_t pad_bottom = p.pad_bottom;
    int32_t pad_left = p.pad_left;
    int32_t pad_right = p.pad_right;
    
    if (p.padding_mode == PaddingMode::SAME) {
        auto h_padding = compute_same_padding(
            impl_->input_height, p.kernel_height, p.stride_height, p.dilation_height);
        auto w_padding = compute_same_padding(
            impl_->input_width, p.kernel_width, p.stride_width, p.dilation_width);
        
        pad_top = h_padding.first;
        pad_bottom = h_padding.second;
        pad_left = w_padding.first;
        pad_right = w_padding.second;
    }
    
    // Main convolution loop
    for (int32_t b = 0; b < impl_->batch_size; ++b) {
        for (int32_t oh = 0; oh < impl_->output_height; ++oh) {
            for (int32_t ow = 0; ow < impl_->output_width; ++ow) {
                for (int32_t oc = 0; oc < p.output_channels; ++oc) {
                    
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int32_t kh = 0; kh < p.kernel_height; ++kh) {
                        for (int32_t kw = 0; kw < p.kernel_width; ++kw) {
                            int32_t ih = oh * p.stride_height + kh * p.dilation_height - pad_top;
                            int32_t iw = ow * p.stride_width + kw * p.dilation_width - pad_left;
                            
                            if (ih >= 0 && ih < impl_->input_height && 
                                iw >= 0 && iw < impl_->input_width) {
                                
                                for (int32_t ic = 0; ic < p.input_channels; ++ic) {
                                    int32_t input_idx = b * impl_->input_height * impl_->input_width * p.input_channels +
                                                       ih * impl_->input_width * p.input_channels +
                                                       iw * p.input_channels + ic;
                                    
                                    int32_t weight_idx = oc * p.kernel_height * p.kernel_width * p.input_channels +
                                                        kh * p.kernel_width * p.input_channels +
                                                        kw * p.input_channels + ic;
                                    
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int32_t output_idx = b * impl_->output_height * impl_->output_width * p.output_channels +
                                        oh * impl_->output_width * p.output_channels +
                                        ow * p.output_channels + oc;
                    
                    output[output_idx] = sum;
                }
            }
        }
    }
    
    // Apply activation function
    if (p.activation != ActivationType::NONE) {
        int32_t output_size = impl_->batch_size * impl_->output_height * 
                             impl_->output_width * p.output_channels;
        apply_activation(output, output_size, p.activation);
    }
    
    return KernelStatus::SUCCESS;
}

KernelStatus CmxConv2D::run_nchw(
    const void* const* inputs,
    void* const* outputs
) {
    const float* input = static_cast<const float*>(inputs[0]);
    const float* weights = static_cast<const float*>(inputs[1]);
    const float* bias = impl_->params.use_bias ? static_cast<const float*>(inputs[2]) : nullptr;
    float* output = static_cast<float*>(outputs[0]);
    
    const Conv2DParams& p = impl_->params;
    
    // Compute padding
    int32_t pad_top = p.pad_top;
    int32_t pad_bottom = p.pad_bottom;
    int32_t pad_left = p.pad_left;
    int32_t pad_right = p.pad_right;
    
    if (p.padding_mode == PaddingMode::SAME) {
        auto h_padding = compute_same_padding(
            impl_->input_height, p.kernel_height, p.stride_height, p.dilation_height);
        auto w_padding = compute_same_padding(
            impl_->input_width, p.kernel_width, p.stride_width, p.dilation_width);
        
        pad_top = h_padding.first;
        pad_bottom = h_padding.second;
        pad_left = w_padding.first;
        pad_right = w_padding.second;
    }
    
    // Main convolution loop for NCHW format
    for (int32_t b = 0; b < impl_->batch_size; ++b) {
        for (int32_t oc = 0; oc < p.output_channels; ++oc) {
            for (int32_t oh = 0; oh < impl_->output_height; ++oh) {
                for (int32_t ow = 0; ow < impl_->output_width; ++ow) {
                    
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int32_t ic = 0; ic < p.input_channels; ++ic) {
                        for (int32_t kh = 0; kh < p.kernel_height; ++kh) {
                            for (int32_t kw = 0; kw < p.kernel_width; ++kw) {
                                int32_t ih = oh * p.stride_height + kh * p.dilation_height - pad_top;
                                int32_t iw = ow * p.stride_width + kw * p.dilation_width - pad_left;
                                
                                if (ih >= 0 && ih < impl_->input_height && 
                                    iw >= 0 && iw < impl_->input_width) {
                                    
                                    int32_t input_idx = b * p.input_channels * impl_->input_height * impl_->input_width +
                                                       ic * impl_->input_height * impl_->input_width +
                                                       ih * impl_->input_width + iw;
                                    
                                    int32_t weight_idx = oc * p.kernel_height * p.kernel_width * p.input_channels +
                                                        kh * p.kernel_width * p.input_channels +
                                                        kw * p.input_channels + ic;
                                    
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    
                    int32_t output_idx = b * p.output_channels * impl_->output_height * impl_->output_width +
                                        oc * impl_->output_height * impl_->output_width +
                                        oh * impl_->output_width + ow;
                    
                    output[output_idx] = sum;
                }
            }
        }
    }
    
    // Apply activation function
    if (p.activation != ActivationType::NONE) {
        int32_t output_size = impl_->batch_size * p.output_channels * 
                             impl_->output_height * impl_->output_width;
        apply_activation(output, output_size, p.activation);
    }
    
    return KernelStatus::SUCCESS;
}

KernelStatus CmxConv2D::run_quantized(
    const void* const* inputs,
    void* const* outputs
) {
    // Quantized implementation would go here
    // This is a placeholder for the quantized convolution path
    return KernelStatus::UNSUPPORTED_DTYPE;
}

// Register the kernel
REGISTER_KERNEL(CmxConv2D, KernelType::CONV2D, 0);

} // namespace kernels
} // namespace cmx