#include "cmx_op_executor.hpp"
#include "cmx_op_registry.hpp"
#include <cstring>
#include <algorithm>

namespace cmx {

// Simple memory pool for scratch allocation
struct memory_pool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    size_t alignment;
    bool is_initialized;
};

// Simple thread pool structure (placeholder)
struct thread_pool {
    uint32_t thread_count;
    bool is_initialized;
};

// Constructor
cmx_op_executor::cmx_op_executor() 
    : config_(cmx_default_executor_config())
    , stats_{}
    , thread_pool_(nullptr)
    , thread_pool_initialized_(false)
    , scratch_pool_(nullptr)
    , scratch_pool_initialized_(false) {
}

cmx_op_executor::cmx_op_executor(const cmx_executor_config& config)
    : config_(config)
    , stats_{}
    , thread_pool_(nullptr)
    , thread_pool_initialized_(false)
    , scratch_pool_(nullptr)
    , scratch_pool_initialized_(false) {
    
    if (config_.max_threads > 1) {
        init_thread_pool(config_.max_threads);
    }
    
    if (config_.scratch_pool_size > 0) {
        init_scratch_pool(config_.scratch_pool_size);
    }
}

// Destructor
cmx_op_executor::~cmx_op_executor() {
    shutdown_thread_pool();
    shutdown_scratch_pool();
}

// Execute a single operation by name
cmx_status cmx_op_executor::execute_op(const std::string& op_name, cmx_op_context& ctx) {
    const cmx_op* op = cmx_get_op(op_name);
    if (!op) {
        return cmx_status::ERROR_UNSUPPORTED_OP;
    }
    
    return execute_op(op, ctx);
}

// Execute a single operation
cmx_status cmx_op_executor::execute_op(const cmx_op* op, cmx_op_context& ctx) {
    if (!op || !op->execute) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Validation if enabled
    if (config_.enable_validation) {
        cmx_status val_status = validate_execution(op, ctx);
        if (val_status != cmx_status::SUCCESS) {
            return val_status;
        }
    }
    
    // Set operation info in context
    ctx.op_name = op->name.c_str();
    
    // Start profiling if enabled
    if (config_.enable_profiling) {
        cmx_start_profiling(&ctx);
    }
    
    // Execute the operation
    cmx_status status = execute_op_internal(op, ctx);
    
    // End profiling if enabled
    if (config_.enable_profiling) {
        cmx_end_profiling(&ctx);
        update_stats(status, cmx_get_execution_time(&ctx));
    }
    
    return status;
}

// Batch execution
cmx_status cmx_op_executor::execute_ops(const std::string* op_names, 
                                       cmx_op_context* contexts, 
                                       size_t count) {
    if (!op_names || !contexts || count == 0) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    for (size_t i = 0; i < count; ++i) {
        cmx_status status = execute_op(op_names[i], contexts[i]);
        if (status != cmx_status::SUCCESS) {
            return status;
        }
    }
    
    return cmx_status::SUCCESS;
}

// Configuration
void cmx_op_executor::set_config(const cmx_executor_config& config) {
    config_ = config;
    
    // Reinitialize resources if needed
    if (config_.max_threads > 1 && !thread_pool_initialized_) {
        init_thread_pool(config_.max_threads);
    }
    
    if (config_.scratch_pool_size > 0 && !scratch_pool_initialized_) {
        init_scratch_pool(config_.scratch_pool_size);
    }
}

const cmx_executor_config& cmx_op_executor::get_config() const {
    return config_;
}

// Statistics
const cmx_exec_stats& cmx_op_executor::get_stats() const {
    return stats_;
}

void cmx_op_executor::reset_stats() {
    std::memset(&stats_, 0, sizeof(stats_));
}

// Thread pool management
cmx_status cmx_op_executor::init_thread_pool(uint32_t thread_count) {
    if (thread_pool_initialized_) {
        shutdown_thread_pool();
    }
    
    thread_pool_ = new thread_pool;
    thread_pool_->thread_count = thread_count;
    thread_pool_->is_initialized = true;
    thread_pool_initialized_ = true;
    
    return cmx_status::SUCCESS;
}

void cmx_op_executor::shutdown_thread_pool() {
    if (thread_pool_initialized_ && thread_pool_) {
        delete thread_pool_;
        thread_pool_ = nullptr;
        thread_pool_initialized_ = false;
    }
}

// Memory pool management
cmx_status cmx_op_executor::init_scratch_pool(size_t pool_size) {
    if (scratch_pool_initialized_) {
        shutdown_scratch_pool();
    }
    
    scratch_pool_ = new memory_pool;
    scratch_pool_->base_ptr = std::aligned_alloc(32, pool_size);
    if (!scratch_pool_->base_ptr) {
        delete scratch_pool_;
        scratch_pool_ = nullptr;
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    scratch_pool_->total_size = pool_size;
    scratch_pool_->used_size = 0;
    scratch_pool_->alignment = 32;
    scratch_pool_->is_initialized = true;
    scratch_pool_initialized_ = true;
    
    return cmx_status::SUCCESS;
}

void cmx_op_executor::shutdown_scratch_pool() {
    if (scratch_pool_initialized_ && scratch_pool_) {
        if (scratch_pool_->base_ptr) {
            std::free(scratch_pool_->base_ptr);
        }
        delete scratch_pool_;
        scratch_pool_ = nullptr;
        scratch_pool_initialized_ = false;
    }
}

void* cmx_op_executor::allocate_scratch(size_t size, size_t alignment) {
    if (!scratch_pool_initialized_ || !scratch_pool_) {
        return std::aligned_alloc(alignment, size);
    }
    
    // Simple bump allocator
    size_t aligned_used = (scratch_pool_->used_size + alignment - 1) & ~(alignment - 1);
    if (aligned_used + size > scratch_pool_->total_size) {
        return nullptr; // Pool exhausted
    }
    
    void* ptr = static_cast<char*>(scratch_pool_->base_ptr) + aligned_used;
    scratch_pool_->used_size = aligned_used + size;
    
    return ptr;
}

void cmx_op_executor::free_scratch(void* ptr) {
    // Simple implementation - in practice, this would be more sophisticated
    if (!scratch_pool_initialized_) {
        std::free(ptr);
    }
    // For pool allocation, we don't free individual chunks in this simple implementation
}

// Internal execution methods
cmx_status cmx_op_executor::execute_op_internal(const cmx_op* op, cmx_op_context& ctx) {
    // Set backend and execution policy defaults if not set
    if (ctx.backend == cmx_backend_type::CPU_SCALAR && 
        config_.default_backend != cmx_backend_type::CPU_SCALAR) {
        ctx.backend = config_.default_backend;
    }
    
    if (ctx.exec_policy == cmx_exec_policy::SERIAL && 
        config_.default_policy != cmx_exec_policy::SERIAL) {
        ctx.exec_policy = config_.default_policy;
    }
    
    // Choose execution path based on policy
    switch (ctx.exec_policy) {
        case cmx_exec_policy::PARALLEL:
            return execute_parallel(op, ctx);
        case cmx_exec_policy::SERIAL:
        default:
            return execute_serial(op, ctx);
    }
}

cmx_status cmx_op_executor::validate_execution(const cmx_op* op, cmx_op_context& ctx) {
    if (!op || !op->execute) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Validate input/output counts
    if (ctx.input_count < op->input_count || ctx.output_count < op->output_count) {
        return cmx_status::ERROR_TENSOR_MISMATCH;
    }
    
    // Validate context
    if (!cmx_validate_context(&ctx)) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    return cmx_status::SUCCESS;
}

void cmx_op_executor::update_stats(cmx_status status, uint64_t execution_time) {
    stats_.total_ops++;
    stats_.total_execution_time += execution_time;
    
    if (status == cmx_status::SUCCESS) {
        stats_.successful_ops++;
    } else {
        stats_.failed_ops++;
    }
    
    stats_.avg_execution_time = stats_.total_ops > 0 ? 
        stats_.total_execution_time / stats_.total_ops : 0;
}

cmx_status cmx_op_executor::execute_parallel(const cmx_op* op, cmx_op_context& ctx) {
    // For now, fall back to serial execution
    // In a full implementation, this would distribute work across threads
    return execute_serial(op, ctx);
}

cmx_status cmx_op_executor::execute_serial(const cmx_op* op, cmx_op_context& ctx) {
    return op->execute(ctx);
}

// C-style API functions
cmx_status cmx_execute_op(const std::string& op_name, cmx_op_context& ctx) {
    return cmx_get_global_executor()->execute_op(op_name, ctx);
}

cmx_status cmx_execute_op_with_config(const std::string& op_name, 
                                     cmx_op_context& ctx,
                                     const cmx_executor_config& config) {
    cmx_op_executor executor(config);
    return executor.execute_op(op_name, ctx);
}

// Global executor instance
cmx_op_executor* cmx_get_global_executor() {
    static cmx_op_executor global_executor;
    return &global_executor;
}

void cmx_set_global_executor_config(const cmx_executor_config& config) {
    cmx_get_global_executor()->set_config(config);
}

// Utility functions
cmx_executor_config cmx_default_executor_config() {
    cmx_executor_config config;
    config.enable_profiling = false;
    config.enable_validation = true;
    config.enable_caching = false;
    config.max_threads = 1;
    config.scratch_pool_size = 0;
    config.default_backend = cmx_backend_type::CPU_SCALAR;
    config.default_policy = cmx_exec_policy::SERIAL;
    return config;
}

const char* cmx_backend_type_to_string(cmx_backend_type backend) {
    switch (backend) {
        case cmx_backend_type::CPU_SCALAR: return "CPU_SCALAR";
        case cmx_backend_type::CPU_SIMD: return "CPU_SIMD";
        case cmx_backend_type::GPU_OPENCL: return "GPU_OPENCL";
        case cmx_backend_type::GPU_VULKAN: return "GPU_VULKAN";
        case cmx_backend_type::DSP: return "DSP";
        case cmx_backend_type::NPU: return "NPU";
        case cmx_backend_type::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

const char* cmx_exec_policy_to_string(cmx_exec_policy policy) {
    switch (policy) {
        case cmx_exec_policy::SERIAL: return "SERIAL";
        case cmx_exec_policy::PARALLEL: return "PARALLEL";
        case cmx_exec_policy::ASYNC: return "ASYNC";
        case cmx_exec_policy::PIPELINE: return "PIPELINE";
        default: return "UNKNOWN";
    }
}

} // namespace cmx