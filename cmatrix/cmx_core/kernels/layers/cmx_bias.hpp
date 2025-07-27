#pragma once

#include <cstdint>

namespace cmx {
namespace kernels {

/**
 * @brief Bias addition layer for element-wise bias application
 * 
 * Adds bias values to input tensors. Used when bias is not fused
 * into other operations like convolution or dense layers.
 */
class CmxBias {
public:
    /**
     * @brief Bias application mode
     */
    enum class BiasMode {
        CHANNEL_WISE,   ///< Apply bias per channel (broadcasting)
        ELEMENT_WISE    ///< Apply bias element-wise (same shape)
    };

    /**
     * @brief Data layout format
     */
    enum class DataLayout {
        NHWC,       ///< Batch, Height, Width, Channels
        NCHW,       ///< Batch, Channels, Height, Width
        NC          ///< Batch, Channels (for 2D tensors)
    };

    /**
     * @brief Configuration structure for Bias layer
     */
    struct Config {
        BiasMode bias_mode;
        DataLayout data_layout;
        int32_t channels;           ///< Number of channels for channel-wise bias
        bool fused_activation;      ///< Whether to apply activation after bias
        int32_t activation_type;    ///< 0: none, 1: relu, 2: relu6, 3: tanh
    };

    /**
     * @brief Tensor shape structure
     */
    struct TensorShape {
        int32_t batch;
        int32_t height;
        int32_t width;
        int32_t channels;
        
        // Constructor for 2D tensors
        TensorShape(int32_t b, int32_t c) : batch(b), height(1), width(1), channels(c) {}
        
        // Constructor for 4D tensors
        TensorShape(int32_t b, int32_t h, int32_t w, int32_t c) 
            : batch(b), height(h), width(w), channels(c) {}
        
        TensorShape() : batch(0), height(0), width(0), channels(0) {}
    };

    CmxBias();
    ~CmxBias();

    /**
     * @brief Configure the Bias layer
     * @param config Configuration parameters
     * @param bias_data Pointer to bias data
     * @return Success status
     */
    bool configure(const Config& config, const float* bias_data);

    /**
     * @brief Execute the Bias operation
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
     * @param weight_memory Memory needed for bias data (bytes)
     * @param workspace_memory Memory needed for workspace (bytes)
     * @return Success status
     */
    bool get_memory_requirements(const TensorShape& input_shape,
                                size_t& weight_memory, size_t& workspace_memory);

private:
    Config config_;
    const float* bias_data_;
    bool is_configured_;

    // Helper methods
    void apply_activation(float* data, size_t size);
    
    // Bias application implementations
    void apply_channel_wise_bias_nhwc(const float* input, const TensorShape& input_shape,
                                     float* output, const TensorShape& output_shape);
    void apply_channel_wise_bias_nchw(const float* input, const TensorShape& input_shape,
                                     float* output, const TensorShape& output_shape);
    void apply_element_wise_bias(const float* input, const TensorShape& input_shape,
                                float* output, const TensorShape& output_shape);
    
    // Optimized implementations
    void apply_bias_vectorized(const float* input, float* output, size_t size, 
                              const float* bias, size_t bias_size, size_t stride);
};

} // namespace kernels
} // namespace cmx