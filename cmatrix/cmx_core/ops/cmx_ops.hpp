#ifndef CMX_OPS_HPP
#define CMX_OPS_HPP

#include <string>
#include <cstdint>

namespace cmx {

// Forward declarations
struct cmx_op_context;
enum class cmx_status : uint8_t;

// Operation metadata and execution function pointer
struct cmx_op {
    std::string name;
    cmx_status (*execute)(cmx_op_context& ctx);
    uint32_t input_count;
    uint32_t output_count;
    uint32_t attr_count;
    bool supports_inplace;
    uint32_t version;
};

// Core operation types
enum class cmx_op_type : uint8_t {
    CONV2D,
    RELU,
    DENSE,
    ADD,
    SUB,
    MUL,
    DIV,
    MAXPOOL2D,
    AVGPOOL2D,
    BATCHNORM,
    SOFTMAX,
    RESHAPE,
    TRANSPOSE,
    CONCAT,
    SPLIT,
    CUSTOM
};

// Operation attributes structure
struct cmx_op_attr {
    enum class type : uint8_t { INT32, FLOAT32, STRING, BOOL } attr_type;
    union {
        int32_t i32_val;
        float f32_val;
        bool bool_val;
        const char* str_val;
    } value;
};

// Status codes
enum class cmx_status : uint8_t {
    SUCCESS = 0,
    ERROR_INVALID_ARGS,
    ERROR_OUT_OF_MEMORY,
    ERROR_UNSUPPORTED_OP,
    ERROR_EXECUTION_FAILED,
    ERROR_INVALID_CONTEXT,
    ERROR_TENSOR_MISMATCH
};

// Core operation implementations
cmx_status cmx_conv2d_execute(cmx_op_context& ctx);
cmx_status cmx_relu_execute(cmx_op_context& ctx);
cmx_status cmx_dense_execute(cmx_op_context& ctx);
cmx_status cmx_add_execute(cmx_op_context& ctx);
cmx_status cmx_sub_execute(cmx_op_context& ctx);
cmx_status cmx_mul_execute(cmx_op_context& ctx);
cmx_status cmx_div_execute(cmx_op_context& ctx);
cmx_status cmx_maxpool2d_execute(cmx_op_context& ctx);
cmx_status cmx_avgpool2d_execute(cmx_op_context& ctx);
cmx_status cmx_batchnorm_execute(cmx_op_context& ctx);
cmx_status cmx_softmax_execute(cmx_op_context& ctx);
cmx_status cmx_reshape_execute(cmx_op_context& ctx);
cmx_status cmx_transpose_execute(cmx_op_context& ctx);
cmx_status cmx_concat_execute(cmx_op_context& ctx);
cmx_status cmx_split_execute(cmx_op_context& ctx);

// Utility functions
const char* cmx_status_to_string(cmx_status status);
cmx_op_type cmx_string_to_op_type(const std::string& name);
const char* cmx_op_type_to_string(cmx_op_type type);

// Core operations initialization
void cmx_init_core_ops();

} // namespace cmx

#endif // CMX_OPS_HPP