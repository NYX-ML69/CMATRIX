#pragma once

#include <cstdint>
#include <cstring>

namespace cmx {
namespace kernels {

/**
 * @brief Memory-efficient vanilla RNN implementation for embedded systems
 * 
 * This class implements a basic recurrent neural network (RNN) cell optimized for
 * microcontrollers and embedded SoCs. It provides the simplest form of recurrent
 * processing, making it suitable for basic time-series models where computational
 * resources are extremely limited.
 * 
 * Features:
 * - Vanilla RNN with configurable activation (ReLU/tanh)
 * - Stateful execution for sequential processing
 * - Minimal memory footprint
 * - No dynamic memory allocation during inference
 * - Optimized for simple time-series tasks
 */
class CmxRNN {
public:
    /**
     * @brief RNN activation function types
     */
    enum class ActivationType {
        TANH,                           ///< Hyperbolic tangent activation
        RELU,                           ///< Rectified linear unit activation
        NONE                            ///< No activation (linear)
    };

    /**
     * @brief RNN layer configuration
     */
    struct Config {
        uint32_t input_size;            ///< Input feature dimension
        uint32_t hidden_size;           ///< Hidden state dimension
        uint32_t batch_size;            ///< Batch size (typically 1 for embedded)
        bool return_sequences;          ///< Return all timesteps or just last
        bool return_state;              ///< Return final state separately
        bool stateful;                  ///< Maintain state between calls
        ActivationType activation;      ///< Activation function type
        
        const float* input_weights;     ///< Input-to-hidden weights [input_size x hidden_size]
        const float* hidden_weights;    ///< Hidden-to-hidden weights [hidden_size x hidden_size]
        const float* bias;              ///< Hidden bias [hidden_size]
    };

    /**
     * @brief RNN internal state
     */
    struct State {
        float* hidden_state;            ///< Hidden state [batch_size x hidden_size]
        bool initialized;               ///< State initialization flag
    };

    /**
     * @brief Constructor
     */
    CmxRNN();

    /**
     * @brief Destructor
     */
    ~CmxRNN();

    /**
     * @brief Configure the RNN layer
     * 
     * @param config Layer configuration parameters
     * @param state_buffer Pre-allocated buffer for internal states
     * @param state_buffer_size Size of the state buffer in bytes
     * @return true if configuration successful, false otherwise
     */
    bool configure(const Config& config, void* state_buffer, size_t state_buffer_size);

    /**
     * @brief Run RNN inference
     * 
     * @param input Input tensor [batch_size x sequence_length x input_size]
     * @param sequence_length Length of input sequence
     * @param output Output tensor [batch_size x (sequence_length or 1) x hidden_size]
     * @param final_state Final hidden state (optional, only if return_state is true)
     * @return true if inference successful, false otherwise
     */
    bool run(const float* input, uint32_t sequence_length, float* output, float* final_state = nullptr);

    /**
     * @brief Infer output shape based on input shape
     * 
     * @param input_shape Input tensor shape [batch, sequence, features]
     * @param output_shape Output tensor shape [batch, sequence or 1, hidden]
     * @return true if shape inference successful, false otherwise
     */
    bool infer_shape(const uint32_t* input_shape, uint32_t* output_shape);

    /**
     * @brief Reset internal state
     */
    void reset_state();

    /**
     * @brief Get required state buffer size
     * 
     * @return Size in bytes needed for state buffer
     */
    size_t get_state_buffer_size() const;

    /**
     * @brief Get current hidden state (read-only)
     * 
     * @return Pointer to hidden state or nullptr if not configured
     */
    const float* get_hidden_state() const;

    /**
     * @brief Set initial hidden state
     * 
     * @param initial_state Initial hidden state [batch_size x hidden_size]
     * @return true if successful, false otherwise
     */
    bool set_initial_state(const float* initial_state);

    /**
     * @brief Set activation function
     * 
     * @param activation New activation function type
     */
    void set_activation(ActivationType activation);

private:
    Config config_;                     ///< Layer configuration
    State state_;                       ///< Internal state
    void* state_buffer_;                ///< Pre-allocated state buffer
    bool configured_;                   ///< Configuration status

    // Temporary buffer for computations
    float* temp_output_;                ///< Temporary output buffer

    /**
     * @brief Compute RNN step: h_t = activation(W_ih * x_t + W_hh * h_{t-1} + b)
     * 
     * @param input Current input vector [input_size]
     * @param prev_hidden Previous hidden state [hidden_size]
     * @param output Current hidden state output [hidden_size]
     */
    void compute_step(const float* input, const float* prev_hidden, float* output);

    /**
     * @brief Apply activation function in-place
     * 
     * @param data Data to apply activation to
     * @param size Number of elements
     */
    void apply_activation(float* data, uint32_t size);

    /**
     * @brief Apply ReLU activation in-place
     * 
     * @param data Data to apply ReLU to
     * @param size Number of elements
     */
    void relu_inplace(float* data, uint32_t size);

    /**
     * @brief Apply tanh activation in-place
     * 
     * @param data Data to apply tanh to
     * @param size Number of elements
     */
    void tanh_inplace(float* data, uint32_t size);

    /**
     * @brief Matrix-vector multiplication: result = matrix * vector + bias
     * 
     * @param matrix Input matrix [rows x cols]
     * @param vector Input vector [cols]
     * @param bias Bias vector [rows] (can be nullptr)
     * @param result Result vector [rows]
     * @param rows Number of matrix rows
     * @param cols Number of matrix columns
     */
    void matvec_multiply(const float* matrix, const float* vector, 
                        const float* bias, float* result, 
                        uint32_t rows, uint32_t cols);

    /**
     * @brief Vector addition: result = a + b
     * 
     * @param a First operand
     * @param b Second operand
     * @param result Result array (can be same as a or b)
     * @param size Number of elements
     */
    void vector_add(const float* a, const float* b, float* result, uint32_t size);

    /**
     * @brief Initialize state buffer layout
     * 
     * @return true if initialization successful, false otherwise
     */
    bool initialize_state_buffer();
};

} // namespace kernels
} // namespace cmx