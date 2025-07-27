#ifndef CMX_OP_CONTEXT_HPP
#define CMX_OP_CONTEXT_HPP

#include <cstdint>
#include <cstddef>

namespace cmx {

// Maximum number of inputs/outputs per operation
constexpr size_t CMX_MAX_OP_INPUTS = 8;
constexpr size_t CMX_MAX_OP_OUTPUTS = 4;
constexpr size_t CMX_MAX_OP_ATTRS = 16;

// Forward declarations
struct cmx_tensor;
struct cmx_op_attr;
enum class cmx_backend_type : uint8_t;

// Backend information
enum class cmx_backend_type : uint8_t {
    CPU_SCALAR,
    CPU_SIMD,
    GPU_OPENCL,
    GPU_VULKAN,
    DSP,
    NPU,
    CUSTOM
};

// Execution policy
enum class cmx_exec_policy : uint8_t {
    SERIAL,
    PARALLEL,
    ASYNC,
    PIPELINE
};

// Tensor descriptor (minimal for bare-metal)
struct cmx_tensor {
    void* data;
    uint32_t* shape;
    uint32_t ndim;
    size_t size;        // Total number of elements
    size_t byte_size;   // Total size in bytes
    uint32_t dtype;     // Data type identifier
    uint32_t stride[4]; // Stride information (max 4D)
    bool is_contiguous;
    uint32_t tensor_id; // Unique identifier
};

// Scratch memory descriptor
struct cmx_scratch_memory {
    void* ptr;
    size_t size;
    size_t alignment;
    bool is_allocated;
};

// Operation context - holds runtime state for one operation
struct cmx_op_context {
    // Input/Output tensors
    cmx_tensor* inputs[CMX_MAX_OP_INPUTS];
    cmx_tensor* outputs[CMX_MAX_OP_OUTPUTS];
    uint32_t input_count;
    uint32_t output_count;
    
    // Operation attributes
    cmx_op_attr* attributes[CMX_MAX_OP_ATTRS];
    uint32_t attr_count;
    
    // Scratch memory for temporary calculations
    cmx_scratch_memory scratch;
    
    // Backend and execution information
    cmx_backend_type backend;
    cmx_exec_policy exec_policy;
    uint32_t thread_count;
    
    // Profiling and debugging
    uint64_t start_time;
    uint64_t end_time;
    uint32_t op_id;
    const char* op_name;
    
    // Error handling
    char error_msg[256];
    uint32_t error_code;
    
    // Memory management flags
    bool owns_inputs;
    bool owns_outputs;
    bool owns_scratch;
};

// Context management functions
cmx_op_context* cmx_create_op_context();
void cmx_destroy_op_context(cmx_op_context* ctx);
void cmx_reset_op_context(cmx_op_context* ctx);

// Input/Output management
cmx_status cmx_set_input(cmx_op_context* ctx, uint32_t index, cmx_tensor* tensor);
cmx_status cmx_set_output(cmx_op_context* ctx, uint32_t index, cmx_tensor* tensor);
cmx_tensor* cmx_get_input(cmx_op_context* ctx, uint32_t index);
cmx_tensor* cmx_get_output(cmx_op_context* ctx, uint32_t index);

// Attribute management
cmx_status cmx_set_attr(cmx_op_context* ctx, uint32_t index, cmx_op_attr* attr);
cmx_op_attr* cmx_get_attr(cmx_op_context* ctx, uint32_t index);
cmx_op_attr* cmx_get_attr_by_name(cmx_op_context* ctx, const char* name);

// Scratch memory management
cmx_status cmx_allocate_scratch(cmx_op_context* ctx, size_t size, size_t alignment = 32);
void cmx_free_scratch(cmx_op_context* ctx);
void* cmx_get_scratch_ptr(cmx_op_context* ctx);

// Backend configuration
void cmx_set_backend(cmx_op_context* ctx, cmx_backend_type backend);
void cmx_set_exec_policy(cmx_op_context* ctx, cmx_exec_policy policy);
void cmx_set_thread_count(cmx_op_context* ctx, uint32_t count);

// Profiling utilities
void cmx_start_profiling(cmx_op_context* ctx);
void cmx_end_profiling(cmx_op_context* ctx);
uint64_t cmx_get_execution_time(cmx_op_context* ctx);

// Error handling
void cmx_set_error(cmx_op_context* ctx, uint32_t error_code, const char* msg);
const char* cmx_get_error_msg(cmx_op_context* ctx);
uint32_t cmx_get_error_code(cmx_op_context* ctx);
void cmx_clear_error(cmx_op_context* ctx);

// Context validation
bool cmx_validate_context(cmx_op_context* ctx);
bool cmx_validate_inputs(cmx_op_context* ctx);
bool cmx_validate_outputs(cmx_op_context* ctx);

} // namespace cmx

#endif // CMX_OP_CONTEXT_HPP