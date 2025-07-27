#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace kernels {

/**
 * @brief Activation function types for fused operations
 */
enum class ActivationType {
    NONE,
    RELU,
    RELU6,
    TANH,
    SIGMOID
};

/**
 * @brief Matrix multiplication layer with optional batch support
 * 
 * Supports standard matrix multiplication with optional fused activation
 * and quantized inputs. Optimized for embedded systems with efficient
 * memory access patterns and weight reordering support.
 */
class CmxMatMul {
public:
    /**
     * @brief Default constructor
     */
    CmxMatMul();

    /**
     * @brief Destructor
     */
    ~CmxMatMul();

    /**
     * @brief Configure matrix multiplication layer
     * 
     * @param batch_size Number of batches (default: 1)
     * @param input_rows Number of input rows (M dimension)
     * @param input_cols Number of input columns (K dimension)
     * @param output_cols Number of output columns (N dimension)
     * @param weights Weight matrix data (K x N)
     * @param bias Optional bias vector (N elements, can be nullptr)
     * @param activation Activation function to apply (default: NONE)
     * @param transpose_weights Whether weights are transposed (default: false)
     * @return true if configuration successful, false otherwise
     */
    bool configure(uint32_t batch_size,
                   uint32_t input_rows, uint32_t input_cols, uint32_t output_cols,
                   const float* weights, const float* bias = nullptr,
                   ActivationType activation = ActivationType::NONE,
                   bool transpose_weights = false);

    /**
     * @brief Execute matrix multiplication
     * 
     * @param input_data Input matrix data (batch_size x input_rows x input_cols)
     * @param output_data Output matrix data (batch_size x input_rows x output_cols)
     * @return true if execution successful, false otherwise
     */
    bool run(const float* input_data, float* output_data);

    /**
     * @brief Infer output shape based on current configuration
     * 
     * @param output_batch Output batch size
     * @param output_rows Output rows
     * @param output_cols Output columns
     * @return true if shape inference successful, false otherwise
     */
    bool infer_shape(uint32_t& output_batch, uint32_t& output_rows,
                     uint32_t& output_cols) const;

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

    /**
     * @brief Set quantization parameters for input
     * 
     * @param input_scale Scale factor for input quantization
     * @param input_zero_point Zero point for input quantization
     * @param weight_scale Scale factor for weight quantization
     * @param weight_zero_point Zero point for weight quantization
     * @param output_scale Scale factor for output quantization
     * @param output_zero_point Zero point for output quantization
     */
    void set_quantization_params(float input_scale, int32_t input_zero_point,
                                float weight_scale, int32_t weight_zero_point,
                                float output_scale, int32_t output_zero_point);

private:
    // Configuration parameters
    uint32_t batch_size_;
    uint32_t input_rows_;
    uint32_t input_cols_;
    uint32_t output_cols_;
    const float* weights_;
    const float* bias_;
    ActivationType activation_;
    bool transpose_weights_;
    
    // Quantization parameters
    bool quantized_;
    float input_scale_;
    int32_t input_zero_point_;
    float weight_scale_;
    int32_t weight_zero_point_;
    float output_scale_;
    int32_t output_zero_point_;
    
    // Configuration state
    bool configured_;
    
    /**
     * @brief Execute standard floating-point matrix multiplication
     * 
     * @param input_data Input matrix data
     * @param output_data Output matrix data
     */
    void execute_float_matmul(const float* input_data, float* output_data);
    
    /**
     * @brief Execute quantized matrix multiplication
     * 
     * @param input_data Input matrix data
     * @param output_data Output matrix data
     */
    void execute_quantized_matmul(const float* input_data, float* output_data);
    
    /**
     * @brief Apply activation function to output
     * 
     * @param value Input value
     * @return Activated value
     */
    float apply_activation(float value) const;
    
    /**
     * @brief Get weight value with optional transposition
     * 
     * @param row Row index
     * @param col Column index
     * @return Weight value
     */
    float get_weight_value(uint32_t row, uint32_t col) const;
    
    /**
     * @brief Add bias if present
     * 
     * @param value Input value
     * @param col Column index
     * @return Value with bias added
     */
    float add_bias(float value, uint32_t col) const;
};

} // namespace kernels
} // namespace cmx