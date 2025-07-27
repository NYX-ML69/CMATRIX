#pragma once

#include <cstdint>
#include <array>

namespace cmx {
namespace kernels {

/**
 * @brief Depthwise Convolution layer for efficient channel-wise filtering
 * 
 * Applies a separate filter to each input channel, reducing computational cost
 * compared to standard convolution. Optimized for mobile and embedded devices.
 */
class CmxDepthwiseConv {
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
        NHWC,       ///< Batch, Height, Width, Channels
        NCHW        ///< Batch, Channels, Height, Width
    };

    /**
     * @brief Configuration structure for DepthwiseConv layer
     */
    struct Config {
        std::array<int32_t, 2> kernel_size;     ///< [height, width]
        std::array<int32_t, 2> strides;         ///< [height_stride, width_stride]
        std::array<int32_t, 2> dilation;        ///< [height_dilation, width_dilation]
        std::array<int32_t, 4> padding;         ///< [top, bottom, left, right]
        PaddingMode padding_mode;
        DataLayout data_layout;
        int32_t input_channels;
        int32_t depth_multiplier;               ///< Number of filters per input channel
        bool use_bias;
        bool fused_activation;
        int32_t activation_type;                ///< 0: none, 1: relu, 2: relu6, 3: tanh
    };

    /**
     * @brief Tensor shape structure
     */
    struct TensorShape {
        int32_t batch;
        int32_t height;
        int32_t width;
        int32_t channels;
    };

    CmxDepthwiseConv();
    ~CmxDepthwiseConv();

    /**
     * @brief Configure the DepthwiseConv layer
     * @param config Configuration parameters
     * @param weights Pointer to depthwise weight data [depth_mult * in_ch][h][w]
     * @param bias Pointer to bias data (optional)
     * @return Success status
     */
    bool configure(const Config& config, const float* weights, const float* bias = nullptr);

    /**
     * @brief Execute the DepthwiseConv operation
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
    
    // Optimized depthwise convolution implementations
    void depthwise_conv_direct(const float* input, const TensorShape& input_shape,
                              float* output, const TensorShape& output_shape);
    void depthwise_conv_simd(const float* input, const TensorShape& input_shape,
                            float* output, const TensorShape& output_shape);
    
    // Channel-wise processing
    void process_channel(const float* input_channel, float* output_channel,
                        int32_t channel_idx, const TensorShape& input_shape,
                        const TensorShape& output_shape);
};

} // namespace kernels
} // namespace cmx