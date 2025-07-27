#pragma once

#include "../kernels/activations/cmx_relu.hpp"
#include "../kernels/layers/cmx_conv2d.hpp"
#include "../kernels/layers/cmx_dense.hpp"
#include "../kernels/layers/cmx_pooling.hpp"
#include "../kernels/math/cmx_gemm.hpp"
#include "../kernels/math/cmx_elementwise.hpp"
#include "cmx_error.hpp"
#include "cmx_types.hpp"

namespace cmx {

/**
 * @brief Kernel function pointer type
 */
typedef cmx_status (*cmx_kernel_func)(const void* params, const cmx_tensor* inputs, 
                                      uint32_t input_count, cmx_tensor* outputs, 
                                      uint32_t output_count);

/**
 * @brief Kernel descriptor structure
 */
struct cmx_kernel_desc {
    const char* name;
    const char* backend;  // cpu, cuda, opencl, etc.
    cmx_kernel_func func;
    uint32_t input_count;
    uint32_t output_count;
    size_t param_size;
};

/**
 * @brief Kernel registry entry
 */
struct cmx_kernel_registry_entry {
    cmx_kernel_desc desc;
    void* lib_handle;  // For dynamically loaded kernels
    bool is_builtin;
};

// Kernel Invocation Macros
#define CMX_INVOKE_KERNEL(kernel_name, params, inputs, input_count, outputs, output_count) \
    cmx_invoke_kernel(#kernel_name, params, inputs, input_count, outputs, output_count)

#define CMX_INVOKE_KERNEL_BACKEND(kernel_name, backend, params, inputs, input_count, outputs, output_count) \
    cmx_invoke_kernel_backend(#kernel_name, #backend, params, inputs, input_count, outputs, output_count)

// Core Kernel Functions
/**
 * @brief Register a kernel implementation
 * @param desc Kernel descriptor
 * @return Status code indicating success or failure
 */
cmx_status cmx_register_kernel(const cmx_kernel_desc& desc);

/**
 * @brief Unregister a kernel
 * @param name Kernel name
 * @param backend Backend name
 * @return Status code indicating success or failure
 */
cmx_status cmx_unregister_kernel(const char* name, const char* backend);

/**
 * @brief Invoke a kernel by name
 * @param kernel_name Name of the kernel
 * @param params Kernel parameters
 * @param inputs Input tensors
 * @param input_count Number of input tensors
 * @param outputs Output tensors
 * @param output_count Number of output tensors
 * @return Status code indicating success or failure
 */
cmx_status cmx_invoke_kernel(const char* kernel_name, const void* params,
                             const cmx_tensor* inputs, uint32_t input_count,
                             cmx_tensor* outputs, uint32_t output_count);

/**
 * @brief Invoke a kernel with specific backend
 * @param kernel_name Name of the kernel
 * @param backend Backend name
 * @param params Kernel parameters
 * @param inputs Input tensors
 * @param input_count Number of input tensors
 * @param outputs Output tensors
 * @param output_count Number of output tensors
 * @return Status code indicating success or failure
 */
cmx_status cmx_invoke_kernel_backend(const char* kernel_name, const char* backend,
                                     const void* params, const cmx_tensor* inputs,
                                     uint32_t input_count, cmx_tensor* outputs,
                                     uint32_t output_count);

/**
 * @brief Get best kernel for given constraints
 * @param kernel_name Name of the kernel
 * @param constraints Performance/hardware constraints
 * @param selected_backend Pointer to store selected backend name
 * @return Status code indicating success or failure
 */
cmx_status cmx_select_best_kernel(const char* kernel_name, 
                                  const cmx_kernel_constraints& constraints,
                                  const char** selected_backend);

/**
 * @brief Load kernels from a dynamic library
 * @param lib_path Path to the library file
 * @return Status code indicating success or failure
 */
cmx_status cmx_load_kernel_library(const char* lib_path);

/**
 * @brief Unload a kernel library
 * @param lib_path Path to the library file
 * @return Status code indicating success or failure
 */
cmx_status cmx_unload_kernel_library(const char* lib_path);

/**
 * @brief Get list of available kernels
 * @param kernel_names Array to store kernel names
 * @param backends Array to store backend names
 * @param max_count Maximum number of entries
 * @param actual_count Pointer to store actual count
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_available_kernels(const char** kernel_names, const char** backends,
                                     uint32_t max_count, uint32_t* actual_count);

/**
 * @brief Check if a kernel is available for a backend
 * @param kernel_name Name of the kernel
 * @param backend Backend name
 * @return true if kernel is available, false otherwise
 */
bool cmx_is_kernel_available(const char* kernel_name, const char* backend);

/**
 * @brief Initialize kernel registry with built-in kernels
 * @return Status code indicating success or failure
 */
cmx_status cmx_init_kernel_registry();

/**
 * @brief Cleanup kernel registry
 * @return Status code indicating success or failure
 */
cmx_status cmx_cleanup_kernel_registry();

/**
 * @brief Benchmark a kernel performance
 * @param kernel_name Name of the kernel
 * @param backend Backend name
 * @param params Kernel parameters
 * @param inputs Input tensors
 * @param input_count Number of input tensors
 * @param outputs Output tensors
 * @param output_count Number of output tensors
 * @param iterations Number of benchmark iterations
 * @param avg_time_ms Pointer to store average execution time
 * @return Status code indicating success or failure
 */
cmx_status cmx_benchmark_kernel(const char* kernel_name, const char* backend,
                                const void* params, const cmx_tensor* inputs,
                                uint32_t input_count, cmx_tensor* outputs,
                                uint32_t output_count, uint32_t iterations,
                                float* avg_time_ms);

} // namespace cmx