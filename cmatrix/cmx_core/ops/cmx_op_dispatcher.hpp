#pragma once

#include <string>
#include <cstdint>

namespace cmx {

// Forward declarations
struct cmx_op;
struct cmx_op_context;
enum class cmx_backend_type : uint8_t;
enum class cmx_tensor_dtype : uint8_t;
enum class cmx_status : uint8_t;

/**
 * @brief Operation dispatch key for kernel selection
 */
struct cmx_dispatch_key {
    std::string op_name;
    cmx_backend_type backend;
    cmx_tensor_dtype input_dtype;
    cmx_tensor_dtype output_dtype;
    uint32_t input_rank;
    uint32_t output_rank;
    
    // Comparison operators for map lookup
    bool operator<(const cmx_dispatch_key& other) const;
    bool operator==(const cmx_dispatch_key& other) const;
};

/**
 * @brief Kernel function signature
 */
using cmx_kernel_fn = cmx_status(*)(const cmx_op_context&);

/**
 * @brief Kernel metadata
 */
struct cmx_kernel_info {
    cmx_kernel_fn kernel;
    const char* name;
    uint32_t priority;  // Higher priority kernels preferred
    bool is_fallback;   // Fallback kernel for unsupported cases
};

/**
 * @brief Operation dispatcher - finds optimal kernel for given context
 */
class cmx_op_dispatcher {
public:
    /**
     * @brief Register a kernel for specific dispatch key
     */
    static cmx_status register_kernel(
        const cmx_dispatch_key& key, 
        const cmx_kernel_info& kernel_info
    );
    
    /**
     * @brief Find best matching kernel for operation context
     */
    static cmx_kernel_fn dispatch_kernel(
        const std::string& op_name,
        const cmx_op_context& context
    );
    
    /**
     * @brief Check if kernel exists for given dispatch key
     */
    static bool has_kernel(const cmx_dispatch_key& key);
    
    /**
     * @brief Get kernel info for debugging/profiling
     */
    static const cmx_kernel_info* get_kernel_info(const cmx_dispatch_key& key);
    
    /**
     * @brief Clear all registered kernels (for testing)
     */
    static void clear_registry();
    
    /**
     * @brief Get total number of registered kernels
     */
    static size_t kernel_count();

private:
    /**
     * @brief Create dispatch key from operation context
     */
    static cmx_dispatch_key create_dispatch_key(
        const std::string& op_name,
        const cmx_op_context& context
    );
    
    /**
     * @brief Find fallback kernel when exact match not found
     */
    static cmx_kernel_fn find_fallback_kernel(
        const std::string& op_name,
        const cmx_op_context& context
    );
    
    /**
     * @brief Validate kernel compatibility with context
     */
    static bool is_kernel_compatible(
        const cmx_dispatch_key& key,
        const cmx_op_context& context
    );
};

/**
 * @brief Macro for automatic kernel registration
 */
#define CMX_REGISTER_KERNEL(op_name, backend, input_dtype, output_dtype, \
                           input_rank, output_rank, kernel_fn, priority) \
    static const bool _cmx_kernel_registered_##kernel_fn = []() { \
        cmx_dispatch_key key{op_name, backend, input_dtype, output_dtype, \
                            input_rank, output_rank}; \
        cmx_kernel_info info{kernel_fn, #kernel_fn, priority, false}; \
        return cmx_op_dispatcher::register_kernel(key, info) == cmx_status::SUCCESS; \
    }()

/**
 * @brief Macro for fallback kernel registration
 */
#define CMX_REGISTER_FALLBACK_KERNEL(op_name, kernel_fn) \
    static const bool _cmx_fallback_registered_##kernel_fn = []() { \
        cmx_dispatch_key key{op_name, cmx_backend_type::CPU, \
                            cmx_tensor_dtype::FLOAT32, cmx_tensor_dtype::FLOAT32, \
                            0, 0}; \
        cmx_kernel_info info{kernel_fn, #kernel_fn, 0, true}; \
        return cmx_op_dispatcher::register_kernel(key, info) == cmx_status::SUCCESS; \
    }()

} // namespace cmx

