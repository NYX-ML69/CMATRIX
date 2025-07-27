#pragma once

#include <cstdint>

namespace cmx {
namespace kernels {

/**
 * @brief Dense (Fully Connected) layer implementation
 * 
 * Performs matrix multiplication between input and weights, with optional bias addition
 * and activation function. Optimized for embedded systems with memory efficiency.
 */
class CmxDense {
public:
    /**
     * @brief Activation function types
     */
    enum class ActivationType {
        NONE = 0,
        RELU = 1,
        RELU6 = 2,
        TANH = 3,
        SIGMOID = 4,
        SOFTMAX = 5
    };

    /**
     * @brief Configuration structure for Dense layer
     */
    struct Config {
        int32_t input_units;        ///< Number of input units
        int32_t output_units;       ///< Number of output units
        bool use_bias;              ///< Whether to use bias
        bool fused_activation;      ///< Whether to apply activation
        ActivationType activation_type;
        bool transpose_weights;     ///< Whether weights are transposed
    };

    /**
     * @brief Tensor shape structure
     */
    struct TensorShape {
        int32_t batch;
        int32_t features;
    };

    CmxDense();
    ~CmxDense();

    /**
     * @brief Configure the Dense layer
     * @param config Configuration parameters
     * @param weights Pointer to weight matrix [input_units x output_units] or [output_units x input_units]
     * @param bias Pointer to bias vector [output_units] (optional)
     * @return Success status
     */
    bool configure(const Config& config, const float* weights, const float* bias = nullptr);

    /**
     * @brief Execute the Dense layer operation
     * @param input Input tensor data [batch x input_units]
     * @param input_shape Input tensor shape
     * @param output Output tensor data [batch x output_units]
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
    void apply_activation(float* data, size_t size);
    void apply_softmax(float* data, size_t batch_size, size_t features);
    
    // Optimized matrix multiplication implementations
    void matmul_naive(const float* input, const TensorShape& input_shape,
                      float* output, const TensorShape& output_shape);
    void matmul_blocked(const float* input, const TensorShape& input_shape,
                       float* output, const TensorShape& output_shape);
    void matmul_simd(const float* input, const TensorShape& input_shape,
                     float* output, const TensorShape& output_shape);
    
    // Vector operations
    void vector_add_bias(float* output, const float* bias, size_t size);
    void vector_relu(float* data, size_t size);
    void vector_relu6(float* data, size_t size);
    void vector_tanh(float* data, size_t size);
    void vector_sigmoid(float* data, size_t size);
};

} // namespace kernels
} // namespace cmx