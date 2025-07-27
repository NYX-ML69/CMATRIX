#pragma once

#include <cstdint>
#include <vector>
#include <memory>

namespace cmx {
namespace kernels {

/**
 * @brief Shape descriptor for tensor dimensions
 */
struct TensorShape {
    std::vector<int32_t> dims;
    int32_t rank() const { return static_cast<int32_t>(dims.size()); }
    int32_t total_size() const {
        int32_t size = 1;
        for (int32_t dim : dims) size *= dim;
        return size;
    }
};

/**
 * @brief Data type enumeration for tensors
 */
enum class DataType {
    FLOAT32,
    INT8,
    INT16,
    INT32,
    UINT8
};

/**
 * @brief Tensor descriptor containing shape and data type information
 */
struct TensorDescriptor {
    TensorShape shape;
    DataType dtype;
    int32_t size_bytes() const;
};

/**
 * @brief Configuration status enumeration
 */
enum class KernelStatus {
    SUCCESS,
    INVALID_PARAMS,
    UNSUPPORTED_DTYPE,
    MEMORY_ERROR,
    SHAPE_MISMATCH
};

/**
 * @brief Pure virtual base class for all kernel implementations
 * 
 * This interface defines the contract that all neural network layer kernels
 * must implement. It provides methods for configuration, execution, and
 * shape inference.
 */
class CmxKernel {
public:
    virtual ~CmxKernel() = default;

    /**
     * @brief Configure the kernel with input/output descriptors and parameters
     * 
     * This method should validate parameters, compute output shapes, and
     * prepare any internal state required for execution. No memory allocation
     * should occur during inference after this call.
     * 
     * @param inputs Vector of input tensor descriptors
     * @param outputs Vector of output tensor descriptors (may be modified)
     * @param params Layer-specific parameters as void pointer
     * @return KernelStatus indicating success or failure
     */
    virtual KernelStatus configure(
        const std::vector<TensorDescriptor>& inputs,
        std::vector<TensorDescriptor>& outputs,
        const void* params
    ) = 0;

    /**
     * @brief Execute the kernel computation
     * 
     * Performs the actual computation using the configured parameters.
     * This method must be thread-safe and should not perform any
     * dynamic memory allocation.
     * 
     * @param inputs Array of input tensor data pointers
     * @param outputs Array of output tensor data pointers
     * @return KernelStatus indicating success or failure
     */
    virtual KernelStatus run(
        const void* const* inputs,
        void* const* outputs
    ) = 0;

    /**
     * @brief Infer output shape from input shapes and parameters
     * 
     * This is a lightweight method that computes output tensor shapes
     * without full kernel configuration. Used for graph compilation
     * and memory planning.
     * 
     * @param input_shapes Vector of input tensor shapes
     * @param params Layer-specific parameters
     * @return Vector of output tensor shapes
     */
    virtual std::vector<TensorShape> infer_shape(
        const std::vector<TensorShape>& input_shapes,
        const void* params
    ) = 0;

    /**
     * @brief Get the kernel type identifier
     * @return String identifier for this kernel type
     */
    virtual const char* get_type() const = 0;

    /**
     * @brief Check if kernel supports the given data type
     * @param dtype Data type to check
     * @return True if supported, false otherwise
     */
    virtual bool supports_dtype(DataType dtype) const = 0;

    /**
     * @brief Get memory requirements for workspace
     * @return Size in bytes of temporary workspace needed
     */
    virtual int32_t get_workspace_size() const { return 0; }

protected:
    /**
     * @brief Validate input tensor count
     * @param inputs Input tensor descriptors
     * @param expected_count Expected number of inputs
     * @return True if count matches, false otherwise
     */
    bool validate_input_count(const std::vector<TensorDescriptor>& inputs, 
                             int32_t expected_count) const {
        return static_cast<int32_t>(inputs.size()) == expected_count;
    }

    /**
     * @brief Validate output tensor count
     * @param outputs Output tensor descriptors
     * @param expected_count Expected number of outputs
     * @return True if count matches, false otherwise
     */
    bool validate_output_count(const std::vector<TensorDescriptor>& outputs, 
                              int32_t expected_count) const {
        return static_cast<int32_t>(outputs.size()) == expected_count;
    }
};

/**
 * @brief Factory function type for kernel creation
 */
using KernelFactory = std::unique_ptr<CmxKernel>(*)();

} // namespace kernels
} // namespace cmx