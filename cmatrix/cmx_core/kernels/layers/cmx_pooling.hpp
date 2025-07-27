#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace kernels {

/**
 * @brief Pooling operation types supported by CmxPooling
 */
enum class PoolingType {
    MAX_POOL,
    AVG_POOL
};

/**
 * @brief Padding mode for pooling operations
 */
enum class PaddingMode {
    VALID,
    SAME
};

/**
 * @brief 2D Pooling layer implementation for embedded ML runtime
 * 
 * Supports both MaxPool and AvgPool operations with configurable
 * window size, stride, and padding. Optimized for memory efficiency
 * and hardware-friendly access patterns.
 */
class CmxPooling {
public:
    /**
     * @brief Default constructor
     */
    CmxPooling();

    /**
     * @brief Destructor
     */
    ~CmxPooling();

    /**
     * @brief Configure pooling layer parameters
     * 
     * @param pool_type Type of pooling operation (MAX_POOL or AVG_POOL)
     * @param window_height Height of pooling window
     * @param window_width Width of pooling window
     * @param stride_height Vertical stride
     * @param stride_width Horizontal stride
     * @param padding Padding mode (VALID or SAME)
     * @param input_height Input tensor height
     * @param input_width Input tensor width
     * @param input_channels Number of input channels
     * @param batch_size Batch size (default: 1)
     * @return true if configuration successful, false otherwise
     */
    bool configure(PoolingType pool_type,
                   uint32_t window_height, uint32_t window_width,
                   uint32_t stride_height, uint32_t stride_width,
                   PaddingMode padding,
                   uint32_t input_height, uint32_t input_width,
                   uint32_t input_channels, uint32_t batch_size = 1);

    /**
     * @brief Execute pooling operation
     * 
     * @param input_data Input tensor data (NHWC format)
     * @param output_data Output tensor data (NHWC format)
     * @return true if execution successful, false otherwise
     */
    bool run(const float* input_data, float* output_data);

    /**
     * @brief Infer output shape based on current configuration
     * 
     * @param output_height Output tensor height
     * @param output_width Output tensor width
     * @param output_channels Output tensor channels
     * @param output_batch Output batch size
     * @return true if shape inference successful, false otherwise
     */
    bool infer_shape(uint32_t& output_height, uint32_t& output_width,
                     uint32_t& output_channels, uint32_t& output_batch) const;

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
    PoolingType pool_type_;
    uint32_t window_height_;
    uint32_t window_width_;
    uint32_t stride_height_;
    uint32_t stride_width_;
    PaddingMode padding_;
    
    // Input/output dimensions
    uint32_t input_height_;
    uint32_t input_width_;
    uint32_t input_channels_;
    uint32_t batch_size_;
    uint32_t output_height_;
    uint32_t output_width_;
    
    // Padding values
    uint32_t pad_top_;
    uint32_t pad_bottom_;
    uint32_t pad_left_;
    uint32_t pad_right_;
    
    // Configuration state
    bool configured_;
    
    /**
     * @brief Calculate padding values based on input dimensions
     */
    void calculate_padding();
    
    /**
     * @brief Execute max pooling operation
     * 
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     */
    void execute_max_pool(const float* input_data, float* output_data);
    
    /**
     * @brief Execute average pooling operation
     * 
     * @param input_data Input tensor data
     * @param output_data Output tensor data
     */
    void execute_avg_pool(const float* input_data, float* output_data);
    
    /**
     * @brief Get input value with padding handling
     * 
     * @param input_data Input tensor data
     * @param batch Batch index
     * @param h Height index
     * @param w Width index
     * @param c Channel index
     * @return Input value or 0 if out of bounds
     */
    float get_input_value(const float* input_data, uint32_t batch,
                         int32_t h, int32_t w, uint32_t c) const;
};

} // namespace kernels
} // namespace cmx