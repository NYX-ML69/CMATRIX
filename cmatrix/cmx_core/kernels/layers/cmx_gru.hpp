#pragma once

#include <cstdint>
#include <cstring>

namespace cmx {
namespace kernels {

/**
 * @brief Memory-efficient GRU cell implementation for embedded systems
 * 
 * This class implements a Gated Recurrent Unit (GRU) cell optimized for
 * microcontrollers and embedded SoCs. It provides similar functionality to
 * LSTM but with fewer parameters and computations, making it ideal for
 * resource-constrained environments.
 * 
 * Features:
 * - Update and reset gates for controlled information flow
 * - Optional return state for time-step chaining
 * - Stateful execution for sequential processing
 * - Quantization-aware precision handling
 * - No dynamic memory allocation during inference
 */
class CmxGRU {
public:
    /**
     * @brief GRU gate configuration
     */
    struct GateConfig {
        const float* input_weights;     ///< Input-to-gate weights [input_size x hidden_size]
        const float* hidden_weights;    ///< Hidden-to-gate weights [hidden_size x hidden_size]
        const float* bias;              ///< Gate bias [hidden_size]
    };

    /**
     * @brief GRU layer configuration
     */
    struct Config {
        uint32_t input_size;            ///< Input feature dimension
        uint32_t hidden_size;           ///< Hidden state dimension
        uint32_t batch_size;            ///< Batch size (typically 1 for embedded)
        bool return_sequences;          ///< Return all timesteps or just last
        bool return_state;              ///< Return final state separately
        bool stateful;                  ///< Maintain state between calls
        
        GateConfig update_gate;         ///< Update gate configuration
        GateConfig reset_gate;          ///< Reset gate configuration
        GateConfig new_gate;            ///< New state candidate gate configuration
    };

    /**
     * @brief GRU internal state
     */
    struct State {
        float* hidden_state;            ///< Hidden state [batch_size x hidden_size]
        bool initialized;               ///< State initialization flag
    };

    /**
     * @brief Constructor
     */
    CmxGRU();

    /**
     * @brief Destructor
     */
    ~CmxGRU();

    /**
     * @brief Configure the GRU layer
     * 
     * @param config Layer configuration parameters
     * @param state_buffer Pre-allocated buffer for internal states
     * @param state_buffer_size Size of the state buffer in bytes
     * @return true if configuration successful, false otherwise
     */
    bool configure(const Config& config, void* state_buffer, size_t state_buffer_size);

    /**
     * @brief Run GRU inference
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

private:
    Config config_;                     ///< Layer configuration
    State state_;                       ///< Internal state
    void* state_buffer_;                ///< Pre-allocated state buffer
    bool configured_;                   ///< Configuration status

    // Temporary buffers for gate computations
    float* temp_update_gate_;           ///< Temporary update gate values
    float* temp_reset_gate_;            ///< Temporary reset gate values
    float* temp_new_gate_;              ///< Temporary new state candidate values
    float* temp_reset_hidden_;          ///< Temporary reset-gated hidden state

    /**
     * @brief Compute single gate activation
     * 
     * @param input Current input vector
     * @param hidden Previous hidden state
     * @param gate_config Gate weights and bias
     * @param output Gate output values
     */
    void compute_gate(const float* input, const float* hidden, 
                      const GateConfig& gate_config, float* output);

    /**
     * @brief Compute new state candidate with reset gate
     * 
     * @param input Current input vector
     * @param hidden Previous hidden state
     * @param reset_gate Reset gate values
     * @param gate_config New gate weights and bias
     * @param output New state candidate values
     */
    void compute_new_state(const float* input, const float* hidden, 
                          const float* reset_gate, const GateConfig& gate_config, 
                          float* output);

    /**
     * @brief Apply sigmoid activation in-place
     * 
     * @param data Data to apply sigmoid to
     * @param size Number of elements
     */
    void sigmoid_inplace(float* data, uint32_t size);

    /**
     * @brief Apply tanh activation in-place
     * 
     * @param data Data to apply tanh to
     * @param size Number of elements
     */
    void tanh_inplace(float* data, uint32_t size);

    /**
     * @brief Element-wise multiplication
     * 
     * @param a First operand
     * @param b Second operand
     * @param result Result array (can be same as a or b)
     * @param size Number of elements
     */
    void elementwise_multiply(const float* a, const float* b, float* result, uint32_t size);

    /**
     * @brief Linear interpolation: result = (1 - alpha) * a + alpha * b
     * 
     * @param a First operand
     * @param b Second operand
     * @param alpha Interpolation weights [size]
     * @param result Result array
     * @param size Number of elements
     */
    void linear_interpolate(const float* a, const float* b, const float* alpha, 
                           float* result, uint32_t size);

    /**
     * @brief Initialize state buffer layout
     * 
     * @return true if initialization successful, false otherwise
     */
    bool initialize_state_buffer();
};

} // namespace kernels
} // namespace cmx