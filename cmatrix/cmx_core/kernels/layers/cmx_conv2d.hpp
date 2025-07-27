#pragma once

#include "cmx_kernel_interface.hpp"
#include <cstdint>

namespace cmx {
namespace kernels {

/**
 * @brief Padding mode enumeration
 */
enum class PaddingMode {
    VALID,    // No padding
    SAME,     // Zero padding to maintain output size
    EXPLICIT  // User-specified padding values
};

/**
 * @brief Data layout format enumeration
 */
enum class DataFormat {
    NHWC,     // Batch, Height, Width, Channel
    NCHW,     // Batch, Channel, Height, Width
    HWC,      // Height, Width, Channel (single batch)
    CHW       // Channel, Height, Width (single batch)
};

/**
 * @brief Activation function enumeration
 */
enum class ActivationType {
    NONE,
    RELU,
    RELU6,
    TANH,
    SIGMOID,
    SWISH
};

/**
 * @brief Configuration parameters for Conv2D layer
 */
struct Conv2DParams {
    // Kernel dimensions
    int32_t kernel_height;
    int32_t kernel_width;
    
    // Stride configuration
    int32_t stride_height;
    int32_t stride_width;
    
    // Dilation configuration
    int32_t dilation_height;
    int32_t dilation_width;
    
    // Padding configuration
    PaddingMode padding_mode;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;
    
    // Channel configuration
    int32_t input_channels;
    int32_t output_channels;
    
    // Data format
    DataFormat data_format;
    
    // Activation function
    ActivationType activation;
    
    // Bias configuration
    bool use_bias;
    
    // Quantization parameters
    bool quantized;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    float* weight_scales;  // Per-channel scales
    
    // Constructor with defaults
    Conv2DParams() 
        : kernel_height(3), kernel_width(3)
        , stride_height(1), stride_width(1)
        , dilation_height(1), dilation_width(1)
        , padding_mode(PaddingMode::VALID)
        , pad_top(0), pad_bottom(0), pad_left(0), pad_right(0)
        , input_channels(0), output_channels(0)
        , data_format(DataFormat::NHWC)
        , activation(ActivationType::NONE)
        , use_bias(false)
        , quantized(false)
        , input_scale(1.0f), input_zero_point(0)
        , output_scale(1.0f), output_zero_point(0)
        , weight_scales(nullptr)
    {}
};

/**
 * @brief 2D Convolution layer implementation
 * 
 * This class implements 2D convolution operations with support for:
 * - Multiple padding modes (VALID, SAME, EXPLICIT)
 * - Configurable strides and dilation
 * - Optional bias addition
 * - Fused activation functions
 * - Quantization-aware inference
 * - Multiple data formats (NHWC, NCHW)
 * - Hardware acceleration when available
 */
class CmxConv2D : public CmxKernel {
public:
    /**
     * @brief Constructor
     */
    CmxConv2D();
    
    /**
     * @brief Destructor
     */
    ~CmxConv2D() override;

    /**
     * @brief Configure the convolution layer
     * 
     * @param inputs Input tensor descriptors (input, weights, [bias])
     * @param outputs Output tensor descriptors (output)
     * @param params Conv2DParams structure
     * @return KernelStatus indicating success or failure
     */
    KernelStatus configure(
        const std::vector<TensorDescriptor>& inputs,
        std::vector<TensorDescriptor>& outputs,
        const void* params
    ) override;

    /**
     * @brief Execute convolution computation
     * 
     * @param inputs Input data pointers (input, weights, [bias])
     * @param outputs Output data pointers (output)
     * @return KernelStatus indicating success or failure
     */
    KernelStatus run(
        const void* const* inputs,
        void* const* outputs
    ) override;

    /**
     * @brief Infer output shape from input shapes
     * 
     * @param input_shapes Input tensor shapes
     * @param params Conv2DParams structure
     * @return Vector containing output tensor shape
     */
    std::vector<TensorShape> infer_shape(
        const std::vector<TensorShape>& input_shapes,
        const void* params
    ) override;

    /**
     * @brief Get kernel type identifier
     * @return "conv2d"
     */
    const char* get_type() const override { return "conv2d"; }

    /**
     * @brief Check data type support
     * @param dtype Data type to check
     * @return True if supported
     */
    bool supports_dtype(DataType dtype) const override;

    /**
     * @brief Get workspace memory requirements
     * @return Workspace size in bytes
     */
    int32_t get_workspace_size() const override;

private:
    struct Conv2DImpl;
    Conv2DImpl* impl_;

    /**
     * @brief Validate configuration parameters
     * @param inputs Input tensor descriptors
     * @param params Conv2D parameters
     * @return KernelStatus indicating validation result
     */
    KernelStatus validate_params(
        const std::vector<TensorDescriptor>& inputs,
        const Conv2DParams& params
    );

    /**
     * @brief Compute output dimensions
     * @param input_height Input tensor height
     * @param input_width Input tensor width
     * @param params Conv2D parameters
     * @return Pair of (output_height, output_width)
     */
    std::pair<int32_t, int32_t> compute_output_size(
        int32_t input_height,
        int32_t input_width,
        const Conv2DParams& params
    );

    /**
     * @brief Compute padding values for SAME mode
     * @param input_size Input dimension size
     * @param kernel_size Kernel dimension size
     * @param stride Stride value
     * @param dilation Dilation value
     * @return Pair of (pad_before, pad_after)
     */
    std::pair<int32_t, int32_t> compute_same_padding(
        int32_t input_size,
        int32_t kernel_size,
        int32_t stride,
        int32_t dilation
    );

    /**
     * @brief Apply activation function
     * @param data Data pointer
     * @param size Number of elements
     * @param activation Activation type
     */
    void apply_activation(float* data, int32_t size, ActivationType activation);

    /**
     * @brief NHWC format convolution implementation
     */
    KernelStatus run_nhwc(
        const void* const* inputs,
        void* const* outputs
    );

    /**
     * @brief NCHW format convolution implementation
     */
    KernelStatus run_nchw(
        const void* const* inputs,
        void* const* outputs
    );

    /**
     * @brief Quantized convolution implementation
     */
    KernelStatus run_quantized(
        const void* const* inputs,
        void* const* outputs
    );
};

} // namespace kernels
} // namespace cmx