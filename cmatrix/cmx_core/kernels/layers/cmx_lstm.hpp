#pragma once

#include <cstdint>
#include <cstring>

namespace cmx {
namespace kernels {

/**
 * @brief Memory-efficient LSTM cell implementation for embedded systems
 * 
 * This class implements a Long Short-Term Memory (LSTM) cell optimized for
 * microcontrollers and embedded SoCs. It supports stateful execution with
 * minimal memory footprint and optional weight fusion for input and hidden states.
 * 
 * Features:
 * - Separate gates: input, forget, output
 * - Weight fusion (input + hidden weights combined)
 * - Stateful execution for sequential processing
 * - Quantization-aware precision handling
 * - No dynamic memory allocation during inference
 */
class CmxLSTM {
public:
    /**
     * @brief LSTM gate configuration
     */
    struct GateConfig {
        const float* input_weights;     ///< Input-to-gate weights [input_size x hidden_size]
        const float* hidden_weights;    ///< Hidden-to-gate weights [hidden_size x hidden_size]
        const float* bias;              ///< Gate bias [hidden_size]
    };

    /**
     * @brief LSTM layer configuration
     */
    struct Config {
        uint32_t input_size;            ///< Input feature dimension
        uint32_t hidden_size;           ///< Hidden state dimension
        uint32_t batch_size;            ///< Batch size (typically 1 for embedded)
        bool return_sequences;          ///< Return all timesteps or just last
        bool stateful;                  ///< Maintain state between calls
        float forget_bias;              ///< Initial forget gate bias
        
        GateConfig input_gate;          ///< Input gate configuration
        GateConfig forget_gate;         ///< Forget gate configuration
        GateConfig output_gate;         ///< Output gate configuration
        GateConfig cell_gate;           ///< Cell candidate gate configuration
    };

    /**
     * @brief LSTM internal state
     */
    struct State {
        float* hidden_state;            ///< Hidden state [batch_size x hidden_size]
        float* cell_state;              ///< Cell state [batch_size x hidden_size]
        bool initialized;               ///< State initialization flag
    };

    /**
     * @brief Constructor
     */
    CmxLSTM();

    /**
     * @brief Destructor
     */
    ~CmxLSTM();

    /**
     * @brief Configure the LSTM layer
     * 
     * @param config Layer configuration parameters
     * @param state_buffer Pre-allocated buffer for internal states
     * @param state_buffer_size Size of the state buffer in bytes
     * @return true if configuration successful, false otherwise
     */
    bool configure(const Config& config, void* state_buffer, size_t state_buffer_size);

    /**
     * @brief Run LSTM inference
     * 
     * @param input Input tensor [batch_size x sequence_length x input_size]
     * @param sequence_length Length of input sequence
     * @param output Output tensor [batch_size x (sequence_length or 1) x hidden_size]
     * @return true if inference successful, false otherwise
     */
    bool run(const float* input, uint32_t sequence_length, float* output);

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
     * @brief Get current cell state (read-only)
     * 
     * @return Pointer to cell state or nullptr if not configured
     */
    const float* get_cell_state() const;

private:
    Config config_;                     ///< Layer configuration
    State state_;                       ///< Internal state
    void* state_buffer_;                ///< Pre-allocated state buffer
    bool configured_;                   ///< Configuration status

    // Temporary buffers for gate computations
    float* temp_input_gate_;            ///< Temporary input gate values
    float* temp_forget_gate_;           ///< Temporary forget gate values
    float* temp_output_gate_;           ///< Temporary output gate values
    float* temp_cell_gate_;             ///< Temporary cell candidate values

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
     * @brief Element-wise multiplication with accumulation
     * 
     * @param a First operand
     * @param b Second operand
     * @param result Result array (can be same as a or b)
     * @param size Number of elements
     */
    void elementwise_multiply(const float* a, const float* b, float* result, uint32_t size);

    /**
     * @brief Element-wise addition
     * 
     * @param a First operand
     * @param b Second operand
     * @param result Result array (can be same as a or b)
     * @param size Number of elements
     */
    void elementwise_add(const float* a, const float* b, float* result, uint32_t size);

    /**
     * @brief Initialize state buffer layout
     * 
     * @return true if initialization successful, false otherwise
     */
    bool initialize_state_buffer();
};

} // namespace kernels
} // namespace cmx