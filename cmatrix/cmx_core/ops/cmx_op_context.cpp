#include "cmx_op_context.hpp"
#include "cmx_ops.hpp"
#include <cstring>
#include <cstdlib>

namespace cmx {

// Context management functions
cmx_op_context* cmx_create_op_context() {
    cmx_op_context* ctx = static_cast<cmx_op_context*>(std::calloc(1, sizeof(cmx_op_context)));
    if (!ctx) {
        return nullptr;
    }
    
    cmx_reset_op_context(ctx);
    return ctx;
}

void cmx_destroy_op_context(cmx_op_context* ctx) {
    if (!ctx) return;
    
    // Free scratch memory if owned
    if (ctx->owns_scratch) {
        cmx_free_scratch(ctx);
    }
    
    std::free(ctx);
}

void cmx_reset_op_context(cmx_op_context* ctx) {
    if (!ctx) return;
    
    // Reset input/output arrays
    std::memset(ctx->inputs, 0, sizeof(ctx->inputs));
    std::memset(ctx->outputs, 0, sizeof(ctx->outputs));
    std::memset(ctx->attributes, 0, sizeof(ctx->attributes));
    
    ctx->input_count = 0;
    ctx->output_count = 0;
    ctx->attr_count = 0;
    
    // Reset scratch memory
    if (ctx->owns_scratch) {
        cmx_free_scratch(ctx);
    }
    std::memset(&ctx->scratch, 0, sizeof(ctx->scratch));
    
    // Reset backend info
    ctx->backend = cmx_backend_type::CPU_SCALAR;
    ctx->exec_policy = cmx_exec_policy::SERIAL;
    ctx->thread_count = 1;
    
    // Reset profiling
    ctx->start_time = 0;
    ctx->end_time = 0;
    ctx->op_id = 0;
    ctx->op_name = nullptr;
    
    // Reset error state
    std::memset(ctx->error_msg, 0, sizeof(ctx->error_msg));
    ctx->error_code = 0;
    
    // Reset ownership flags
    ctx->owns_inputs = false;
    ctx->owns_outputs = false;
    ctx->owns_scratch = false;
}

// Input/Output management
cmx_status cmx_set_input(cmx_op_context* ctx, uint32_t index, cmx_tensor* tensor) {
    if (!ctx || index >= CMX_MAX_OP_INPUTS) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    ctx->inputs[index] = tensor;
    if (index >= ctx->input_count) {
        ctx->input_count = index + 1;
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_set_output(cmx_op_context* ctx, uint32_t index, cmx_tensor* tensor) {
    if (!ctx || index >= CMX_MAX_OP_OUTPUTS) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    ctx->outputs[index] = tensor;
    if (index >= ctx->output_count) {
        ctx->output_count = index + 1;
    }
    
    return cmx_status::SUCCESS;
}

cmx_tensor* cmx_get_input(cmx_op_context* ctx, uint32_t index) {
    if (!ctx || index >= ctx->input_count) {
        return nullptr;
    }
    return ctx->inputs[index];
}

cmx_tensor* cmx_get_output(cmx_op_context* ctx, uint32_t index) {
    if (!ctx || index >= ctx->output_count) {
        return nullptr;
    }
    return ctx->outputs[index];
}

// Attribute management
cmx_status cmx_set_attr(cmx_op_context* ctx, uint32_t index, cmx_op_attr* attr) {
    if (!ctx || index >= CMX_MAX_OP_ATTRS) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    ctx->attributes[index] = attr;
    if (index >= ctx->attr_count) {
        ctx->attr_count = index + 1;
    }
    
    return cmx_status::SUCCESS;
}

cmx_op_attr* cmx_get_attr(cmx_op_context* ctx, uint32_t index) {
    if (!ctx || index >= ctx->attr_count) {
        return nullptr;
    }
    return ctx->attributes[index];
}

cmx_op_attr* cmx_get_attr_by_name(cmx_op_context* ctx, const char* name) {
    // This would require attribute names to be stored - simplified for now
    return nullptr;
}

// Scratch memory management
cmx_status cmx_allocate_scratch(cmx_op_context* ctx, size_t size, size_t alignment) {
    if (!ctx || size == 0) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Free existing scratch memory
    if (ctx->scratch.is_allocated) {
        cmx_free_scratch(ctx);
    }
    
    // Allocate aligned memory
    ctx->scratch.ptr = std::aligned_alloc(alignment, size);
    if (!ctx->scratch.ptr) {
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    ctx->scratch.size = size;
    ctx->scratch.alignment = alignment;
    ctx->scratch.is_allocated = true;
    ctx->owns_scratch = true;
    
    return cmx_status::SUCCESS;
}

void cmx_free_scratch(cmx_op_context* ctx) {
    if (!ctx || !ctx->scratch.is_allocated) {
        return;
    }
    
    std::free(ctx->scratch.ptr);
    ctx->scratch.ptr = nullptr;
    ctx->scratch.size = 0;
    ctx->scratch.alignment = 0;
    ctx->scratch.is_allocated = false;
    ctx->owns_scratch = false;
}

void* cmx_get_scratch_ptr(cmx_op_context* ctx) {
    if (!ctx || !ctx->scratch.is_allocated) {
        return nullptr;
    }
    return ctx->scratch.ptr;
}

// Backend configuration
void cmx_set_backend(cmx_op_context* ctx, cmx_backend_type backend) {
    if (ctx) {
        ctx->backend = backend;
    }
}

void cmx_set_exec_policy(cmx_op_context* ctx, cmx_exec_policy policy) {
    if (ctx) {
        ctx->exec_policy = policy;
    }
}

void cmx_set_thread_count(cmx_op_context* ctx, uint32_t count) {
    if (ctx) {
        ctx->thread_count = count > 0 ? count : 1;
    }
}

// Profiling utilities
void cmx_start_profiling(cmx_op_context* ctx) {
    if (ctx) {
        // Simple timestamp - in real implementation, use high-res timer
        ctx->start_time = 0; // Would get actual timestamp
    }
}

void cmx_end_profiling(cmx_op_context* ctx) {
    if (ctx) {
        ctx->end_time = 0; // Would get actual timestamp
    }
}

uint64_t cmx_get_execution_time(cmx_op_context* ctx) {
    if (!ctx) return 0;
    return ctx->end_time - ctx->start_time;
}

// Error handling
void cmx_set_error(cmx_op_context* ctx, uint32_t error_code, const char* msg) {
    if (!ctx) return;
    
    ctx->error_code = error_code;
    if (msg) {
        std::strncpy(ctx->error_msg, msg, sizeof(ctx->error_msg) - 1);
        ctx->error_msg[sizeof(ctx->error_msg) - 1] = '\0';
    }
}

const char* cmx_get_error_msg(cmx_op_context* ctx) {
    return ctx ? ctx->error_msg : nullptr;
}

uint32_t cmx_get_error_code(cmx_op_context* ctx) {
    return ctx ? ctx->error_code : 0;
}

void cmx_clear_error(cmx_op_context* ctx) {
    if (ctx) {
        ctx->error_code = 0;
        std::memset(ctx->error_msg, 0, sizeof(ctx->error_msg));
    }
}

// Context validation
bool cmx_validate_context(cmx_op_context* ctx) {
    if (!ctx) return false;
    return cmx_validate_inputs(ctx) && cmx_validate_outputs(ctx);
}

bool cmx_validate_inputs(cmx_op_context* ctx) {
    if (!ctx) return false;
    
    for (uint32_t i = 0; i < ctx->input_count; ++i) {
        if (!ctx->inputs[i] || !ctx->inputs[i]->data) {
            return false;
        }
    }
    return true;
}

bool cmx_validate_outputs(cmx_op_context* ctx) {
    if (!ctx) return false;
    
    for (uint32_t i = 0; i < ctx->output_count; ++i) {
        if (!ctx->outputs[i] || !ctx->outputs[i]->data) {
            return false;
        }
    }
    return true;
}

} // namespace cmx