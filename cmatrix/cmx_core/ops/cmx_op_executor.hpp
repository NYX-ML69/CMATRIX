#ifndef CMX_OP_EXECUTOR_HPP
#define CMX_OP_EXECUTOR_HPP

#include "cmx_ops.hpp"
#include "cmx_op_context.hpp"
#include <cstdint>

namespace cmx {

// Execution statistics
struct cmx_exec_stats {
    uint64_t total_ops;
    uint64_t successful_ops;
    uint64_t failed_ops;
    uint64_t total_execution_time;
    uint64_t avg_execution_time;
    uint32_t cache_hits;
    uint32_t cache_misses;
};

// Executor configuration
struct cmx_executor_config {
    bool enable_profiling;
    bool enable_validation;
    bool enable_caching;
    uint32_t max_threads;
    size_t scratch_pool_size;
    cmx_backend_type default_backend;
    cmx_exec_policy default_policy;
};

// Operation executor class
class cmx_op_executor {
public:
    cmx_op_executor();
    explicit cmx_op_executor(const cmx_executor_config& config);
    ~cmx_op_executor();
    
    // Execute a single operation
    cmx_status execute_op(const std::string& op_name, cmx_op_context& ctx);
    cmx_status execute_op(const cmx_op* op, cmx_op_context& ctx);
    
    // Batch execution
    cmx_status execute_ops(const std::string* op_names, cmx_op_context* contexts, 
                          size_t count);
    
    // Configuration
    void set_config(const cmx_executor_config& config);
    const cmx_executor_config& get_config() const;
    
    // Statistics
    const cmx_exec_stats& get_stats() const;
    void reset_stats();
    
    // Thread pool management (for parallel execution)
    cmx_status init_thread_pool(uint32_t thread_count);
    void shutdown_thread_pool();
    
    // Memory pool management
    cmx_status init_scratch_pool(size_t pool_size);
    void shutdown_scratch_pool();
    void* allocate_scratch(size_t size, size_t alignment = 32);
    void free_scratch(void* ptr);
    
private:
    cmx_executor_config config_;
    cmx_exec_stats stats_;
    
    // Thread pool (simple implementation)
    struct thread_pool* thread_pool_;
    bool thread_pool_initialized_;
    
    // Scratch memory pool
    struct memory_pool* scratch_pool_;
    bool scratch_pool_initialized_;
    
    // Internal execution methods
    cmx_status execute_op_internal(const cmx_op* op, cmx_op_context& ctx);
    cmx_status validate_execution(const cmx_op* op, cmx_op_context& ctx);
    void update_stats(cmx_status status, uint64_t execution_time);
    
    // Parallel execution helpers
    cmx_status execute_parallel(const cmx_op* op, cmx_op_context& ctx);
    cmx_status execute_serial(const cmx_op* op, cmx_op_context& ctx);
};

// C-style API functions
cmx_status cmx_execute_op(const std::string& op_name, cmx_op_context& ctx);
cmx_status cmx_execute_op_with_config(const std::string& op_name, 
                                     cmx_op_context& ctx,
                                     const cmx_executor_config& config);

// Global executor instance management
cmx_op_executor* cmx_get_global_executor();
void cmx_set_global_executor_config(const cmx_executor_config& config);

// Utility functions
cmx_executor_config cmx_default_executor_config();
const char* cmx_backend_type_to_string(cmx_backend_type backend);
const char* cmx_exec_policy_to_string(cmx_exec_policy policy);

} // namespace cmx

#endif // CMX_OP_EXECUTOR_HPP