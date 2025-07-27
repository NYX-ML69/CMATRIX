#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace kernels {

/**
 * @brief Normalization layer types
 */
enum class NormalizationType {
    BATCH_NORM,
    LAYER_NORM,
    INSTANCE_NORM
};

/**
 * @brief Post-normalization activation functions
 */
enum class PostNormActivation {
    NONE,
    RELU,
    RELU6,
    TANH,
    SIGMOID
};

/**
 * @brief Normalization layer implementation for embedded ML runtime
 * 
 * Supports BatchNorm, LayerNorm, and InstanceNorm with configurable
 * scale, offset, mean, and variance parameters. Includes fused
 * post-activation support for efficient inference.
 */
class CmxNormalization {
public:
    /**
     * @brief Default constructor
     */
    CmxNormalization();

    /**
     * @brief Destructor
     */
    ~CmxNormalization();

    /**
     * @brief Configure normalization layer
     * 
     * @param norm_type Type of normalization (BATCH_NORM, LAYER_NORM, INSTANCE_NORM)
     * @param batch_size Batch size
     * @param height Input height (for spatial dimensions)
     * @param width Input width (for spatial dimensions)
     * @param channels Number of channels
     * @param scale Scale parameters (gamma)
     * @param offset Offset parameters (beta)
     * @param mean Mean values (for batch norm)
     * @param variance Variance values (for batch norm)
     * @param epsilon Small constant for numerical stability (default: 1e-5)
     * @param activation Post-normalization activation (default: NONE)
     * @return true if configuration successful, false otherwise
     */
    bool configure(NormalizationType norm_type,
                   uint32_t batch_size, uint32_t height, uint32_t width, uint32_t channels,
                   const float* scale, const float* offset,
                   const float* mean = nullptr, const float* variance = nullptr,
                   float epsilon = 1e-5f,
                   PostNormActivation activation = PostNormActivation::NONE);

    /**
     * @brief Execute normalization operation
     * 
     * @param input_data Input tensor data (NHWC format)
     * @param output_data Output tensor data (NHWC format)
     * @return true if execution successful, false otherwise
     */
    bool run(const float* input_data, float* output_data);

    /**
     * @brief Infer output shape (same as input shape)
     * 
     * @param output_batch Output batch size
     * @param output_height Output height
     * @param output_width Output width
     * @param output_channels Output channels
     * @return true if shape inference successful, false otherwise
     */
    bool infer_shape(uint32_t& output_batch, uint32_t& output_height,
                     uint32_t& output_width, uint32_t& output_channels) const;

    /**
     * @brief Get memory requirements for this layer
     * 
     * @return Required memory in bytes
     */
    size_t get_memory_requirements() const;

    /**
     * @brief Check if layer is properly configured
     * 
     * @return true if configured, false otherwise
     */
    bool is_configured() const { return configured_; }

private:
    // Configuration parameters
    NormalizationType norm_type_;
    uint32_t batch_size_;
    uint32_t height_;
    uint32_t width_;
    uint32_t channels_;
    const float* scale_;
    const float* offset_;
    const float* mean_;
    const float* variance_;
    float epsilon_;
    PostNormActivation activation_;
    
    // Configuration state
    bool configured_;
    
    /**
     * @brief Execute batch normalization
     * 
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     */
    void execute_batch_norm(const float* input_data, float* output_data);
    
    /**
     * @brief Execute layer normalization
     * 
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     */
    void execute_layer_norm(const float* input_data, float* output_data);
    
    /**
     * @brief Execute instance normalization
     * 
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     */
    void execute_instance_norm(const float* input_data, float* output_data);
    
    /**
     * @brief Apply post-normalization activation
     * 
     * @param value Input value
     * @return Activated value
     */
    float apply_activation(float value) const;
    
    /**
     * @brief Calculate mean and variance for a subset of data
     * 
     * @param data Input data
     * @param size Number of elements
     * @param mean Output mean
     * @param variance Output variance
     */
    void calculate_stats(const float* data, uint32_t size, float& mean, float& variance) const;
};

} // namespace kernels
} // namespace cmx