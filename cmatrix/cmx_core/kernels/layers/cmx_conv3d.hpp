#pragma once

#include <cstdint>
#include <array>

namespace cmx {
namespace kernels {

/**
 * @brief 3D Convolution layer for volumetric data processing
 * 
 * Supports 5D tensors (NDHWC format) with configurable depth, height, width kernels.
 * Optimized for embedded systems with minimal memory footprint.
 */
class CmxConv3D {
public:
    /**
     * @brief Padding mode for convolution
     */
    enum class PaddingMode {
        VALID,      ///< No padding
        SAME,       ///< Pad to maintain output size
        EXPLICIT    ///< User-defined padding
    };

    /**
     * @brief Data layout format
     */
    enum class DataLayout {
        NDHWC,      ///< Batch, Depth, Height, Width, Channels
        NCDHW       ///< Batch, Channels, Depth, Height, Width
    };

    /**
     * @brief Configuration structure for Conv3D layer
     */
    struct Config {
        std::array<int32_t, 3> kernel_size;     ///< [depth, height, width]
        std::array<int32_t, 3> strides;         ///< [depth_stride, height_stride, width_stride]
        std::array<int32_t, 3> dilation;        ///< [depth_dilation, height_dilation, width_dilation]
        std::array<int32_t, 6> padding;         ///< [d_front, d_back, h_top, h_bottom, w_left, w_right]
        PaddingMode padding_mode;
        DataLayout data_layout;
        int32_t input_channels;
        int32_t output_channels;
        bool use_bias;
        bool fused_activation;
        int32_t activation_type;  ///< 0: none, 1: relu, 2: relu6, 3: tanh
    };

    /**
     * @brief Tensor shape structure
     */
    struct TensorShape {
        int32_t batch;
        int32_t depth;
        int32_t height;
        int32_t width;
        int32_t channels;
    };

    CmxConv3D();
    ~CmxConv3D();

    /**
     * @brief Configure the Conv3D layer
     * @param config Configuration parameters
     * @param weights Pointer to weight data [out_ch][in_ch][d][h][w]
     * @param bias Pointer to bias data (optional)
     * @return Success status
     */
    bool configure(const Config& config, const float* weights, const float* bias = nullptr);

    /**
     * @brief Execute the Conv3D operation
     * @param input Input tensor data
     * @param input_shape Input tensor shape
     * @param output Output tensor data
     * @param output_shape Output tensor shape
     * @return Success status
     */
    bool run(const float* input, const TensorShape& input_shape,
             float* output, const TensorShape& output_shape);

    /**
     * @brief Infer output shape from input shape and configuration
     * @param input_shape Input tensor shape
     * @param output_shape Inferred output tensor shape
     * @return Success status
     */
    bool infer_shape(const TensorShape& input_shape, TensorShape& output_shape);

    /**
     * @brief Get memory requirements for the layer
     * @param input_shape Input tensor shape
     * @param weight_memory Memory needed for weights (bytes)
     * @param workspace_memory Memory needed for workspace (bytes)
     * @return Success status
     */
    bool get_memory_requirements(const TensorShape& input_shape,
                                size_t& weight_memory, size_t& workspace_memory);

private:
    Config config_;
    const float* weights_;
    const float* bias_;
    bool is_configured_;

    // Helper methods
    void calculate_output_size(const TensorShape& input_shape, TensorShape& output_shape);
    void calculate_padding();
    void apply_activation(float* data, size_t size);
    
    // Optimized convolution implementations
    void conv3d_direct(const float* input, const TensorShape& input_shape,
                       float* output, const TensorShape& output_shape);
    void conv3d_im2col(const float* input, const TensorShape& input_shape,
                       float* output, const TensorShape& output_shape);
};

} // namespace kernels
} // namespace cmx