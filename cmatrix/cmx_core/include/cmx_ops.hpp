#pragma once

#include "cmx_ops.hpp"
#include "cmx_op_executor.hpp"
#include "cmx_op_context.hpp"
#include "cmx_op_dispatcher.hpp"
#include "cmx_error.hpp"
#include "cmx_types.hpp"

namespace cmx {

/**
 * @brief Base operation structure
 */
struct cmx_op {
    const char* name;
    const char* type;
    void* params;
    uint32_t param_size;
    cmx_op_executor executor;
};

/**
 * @brief Operation registry entry
 */
struct cmx_op_registry_entry {
    const char* op_name;
    cmx_op_creator creator_func;
    cmx_op_executor executor_func;
    cmx_op_validator validator_func;
};

// Core Operations
/**
 * @brief ReLU activation operation
 */
extern const cmx_op cmx_relu_op;

/**
 * @brief Element-wise addition operation
 */
extern const cmx_op cmx_add_op;

/**
 * @brief Element-wise multiplication operation
 */
extern const cmx_op cmx_mul_op;

/**
 * @brief Matrix multiplication operation
 */
extern const cmx_op cmx_matmul_op;

/**
 * @brief 2D Convolution operation
 */
extern const cmx_op cmx_conv2d_op;

/**
 * @brief Max pooling operation
 */
extern const cmx_op cmx_maxpool_op;

/**
 * @brief Batch normalization operation
 */
extern const cmx_op cmx_batchnorm_op;

/**
 * @brief Softmax operation
 */
extern const cmx_op cmx_softmax_op;

/**
 * @brief Dropout operation
 */
extern const cmx_op cmx_dropout_op;

/**
 * @brief Reshape operation
 */
extern const cmx_op cmx_reshape_op;

// Operation Registration Functions
/**
 * @brief Register a new operation type
 * @param op_name Name of the operation
 * @param entry Registry entry containing operation functions
 * @return Status code indicating success or failure
 */
cmx_status cmx_register_op(const char* op_name, const cmx_op_registry_entry& entry);

/**
 * @brief Unregister an operation type
 * @param op_name Name of the operation to unregister
 * @return Status code indicating success or failure
 */
cmx_status cmx_unregister_op(const char* op_name);

/**
 * @brief Create an operation instance
 * @param op_name Name of the operation type
 * @param params Operation parameters
 * @param param_size Size of parameters in bytes
 * @return Created operation instance
 */
cmx_op cmx_create_op(const char* op_name, const void* params, uint32_t param_size);

/**
 * @brief Destroy an operation instance
 * @param op Pointer to operation to destroy
 * @return Status code indicating success or failure
 */
cmx_status cmx_destroy_op(cmx_op* op);

/**
 * @brief Execute an operation
 * @param op Operation to execute
 * @param context Execution context
 * @return Status code indicating success or failure
 */
cmx_status cmx_execute_op(const cmx_op& op, cmx_op_context& context);

/**
 * @brief Validate operation parameters
 * @param op Operation to validate
 * @return Status code indicating success or failure
 */
cmx_status cmx_validate_op(const cmx_op& op);

/**
 * @brief Get list of registered operations
 * @param op_names Array to store operation names
 * @param max_count Maximum number of names to retrieve
 * @param actual_count Pointer to store actual number of operations
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_registered_ops(const char** op_names, uint32_t max_count, uint32_t* actual_count);

/**
 * @brief Check if an operation is registered
 * @param op_name Name of the operation to check
 * @return true if operation is registered, false otherwise
 */
bool cmx_is_op_registered(const char* op_name);

/**
 * @brief Initialize operation registry with built-in operations
 * @return Status code indicating success or failure
 */
cmx_status cmx_init_op_registry();

/**
 * @brief Cleanup operation registry
 * @return Status code indicating success or failure
 */
cmx_status cmx_cleanup_op_registry();

} // namespace cmx