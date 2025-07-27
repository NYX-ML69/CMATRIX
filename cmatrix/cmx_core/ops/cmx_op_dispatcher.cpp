#include "cmx_op_dispatcher.hpp"
#include "cmx_op_context.hpp"
#include "../core/cmx_tensor.hpp"
#include "../core/cmx_status.hpp"
#include "../core/cmx_backend.hpp"

#include <map>
#include <vector>
#include <algorithm>

namespace cmx {

// Static kernel registry
static std::map<cmx_dispatch_key, cmx_kernel_info> g_kernel_registry;
static std::vector<std::pair<std::string, cmx_kernel_info>> g_fallback_kernels;

bool cmx_dispatch_key::operator<(const cmx_dispatch_key& other) const {
    if (op_name != other.op_name) return op_name < other.op_name;
    if (backend != other.backend) return static_cast<uint8_t>(backend) < static_cast<uint8_t>(other.backend);
    if (input_dtype != other.input_dtype) return static_cast<uint8_t>(input_dtype) < static_cast<uint8_t>(other.input_dtype);
    if (output_dtype != other.output_dtype) return static_cast<uint8_t>(output_dtype) < static_cast<uint8_t>(other.output_dtype);
    if (input_rank != other.input_rank) return input_rank < other.input_rank;
    return output_rank < other.output_rank;
}

bool cmx_dispatch_key::operator==(const cmx_dispatch_key& other) const {
    return op_name == other.op_name &&
           backend == other.backend &&
           input_dtype == other.input_dtype &&
           output_dtype == other.output_dtype &&
           input_rank == other.input_rank &&
           output_rank == other.output_rank;
}

cmx_status cmx_op_dispatcher::register_kernel(
    const cmx_dispatch_key& key, 
    const cmx_kernel_info& kernel_info
) {
    if (!kernel_info.kernel) {
        return cmx_status::INVALID_ARGUMENT;
    }
    
    if (kernel_info.is_fallback) {
        // Store fallback kernels separately
        g_fallback_kernels.emplace_back(key.op_name, kernel_info);
    } else {
        // Check for duplicate registration with lower priority
        auto it = g_kernel_registry.find(key);
        if (it != g_kernel_registry.end()) {
            if (kernel_info.priority <= it->second.priority) {
                return cmx_status::ALREADY_EXISTS;
            }
        }
        
        g_kernel_registry[key] = kernel_info;
    }
    
    return cmx_status::SUCCESS;
}

cmx_kernel_fn cmx_op_dispatcher::dispatch_kernel(
    const std::string& op_name,
    const cmx_op_context& context
) {
    // Create dispatch key from context
    cmx_dispatch_key key = create_dispatch_key(op_name, context);
    
    // Try exact match first
    auto it = g_kernel_registry.find(key);
    if (it != g_kernel_registry.end()) {
        return it->second.kernel;
    }
    
    // Try relaxed matching (ignore ranks for flexible kernels)
    key.input_rank = 0;
    key.output_rank = 0;
    it = g_kernel_registry.find(key);
    if (it != g_kernel_registry.end()) {
        return it->second.kernel;
    }
    
    // Try dtype-agnostic matching
    key.input_dtype = cmx_tensor_dtype::FLOAT32;
    key.output_dtype = cmx_tensor_dtype::FLOAT32;
    it = g_kernel_registry.find(key);
    if (it != g_kernel_registry.end()) {
        return it->second.kernel;
    }
    
    // Fall back to CPU backend if using specialized backend
    if (key.backend != cmx_backend_type::CPU) {
        key.backend = cmx_backend_type::CPU;
        it = g_kernel_registry.find(key);
        if (it != g_kernel_registry.end()) {
            return it->second.kernel;
        }
    }
    
    // Search fallback kernels
    return find_fallback_kernel(op_name, context);
}

bool cmx_op_dispatcher::has_kernel(const cmx_dispatch_key& key) {
    return g_kernel_registry.find(key) != g_kernel_registry.end();
}

const cmx_kernel_info* cmx_op_dispatcher::get_kernel_info(const cmx_dispatch_key& key) {
    auto it = g_kernel_registry.find(key);
    return (it != g_kernel_registry.end()) ? &it->second : nullptr;
}

void cmx_op_dispatcher::clear_registry() {
    g_kernel_registry.clear();
    g_fallback_kernels.clear();
}

size_t cmx_op_dispatcher::kernel_count() {
    return g_kernel_registry.size() + g_fallback_kernels.size();
}

cmx_dispatch_key cmx_op_dispatcher::create_dispatch_key(
    const std::string& op_name,
    const cmx_op_context& context
) {
    cmx_dispatch_key key;
    key.op_name = op_name;
    key.backend = context.backend_type;
    
    // Extract tensor info from first input/output
    if (context.num_inputs > 0 && context.inputs) {
        key.input_dtype = context.inputs[0].dtype;
        key.input_rank = context.inputs[0].rank;
    } else {
        key.input_dtype = cmx_tensor_dtype::FLOAT32;
        key.input_rank = 0;
    }
    
    if (context.num_outputs > 0 && context.outputs) {
        key.output_dtype = context.outputs[0].dtype;
        key.output_rank = context.outputs[0].rank;
    } else {
        key.output_dtype = cmx_tensor_dtype::FLOAT32;
        key.output_rank = 0;
    }
    
    return key;
}

cmx_kernel_fn cmx_op_dispatcher::find_fallback_kernel(
    const std::string& op_name,
    const cmx_op_context& context
) {
    // Find highest priority fallback kernel for this operation
    cmx_kernel_fn best_kernel = nullptr;
    uint32_t best_priority = 0;
    
    for (const auto& fallback : g_fallback_kernels) {
        if (fallback.first == op_name) {
            if (fallback.second.priority > best_priority) {
                best_kernel = fallback.second.kernel;
                best_priority = fallback.second.priority;
            }
        }
    }
    
    return best_kernel;
}

bool cmx_op_dispatcher::is_kernel_compatible(
    const cmx_dispatch_key& key,
    const cmx_op_context& context
) {
    // Basic compatibility checks
    if (key.op_name.empty() || !context.inputs || !context.outputs) {
        return false;
    }
    
    // Check tensor rank compatibility (0 means flexible)
    if (key.input_rank != 0 && context.num_inputs > 0) {
        if (context.inputs[0].rank != key.input_rank) {
            return false;
        }
    }
    
    if (key.output_rank != 0 && context.num_outputs > 0) {
        if (context.outputs[0].rank != key.output_rank) {
            return false;
        }
    }
    
    return true;
}

} // namespace cmx