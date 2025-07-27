#include "cmx_op_registry.hpp"
#include <cstring>
#include <algorithm>

namespace cmx {

// Get singleton instance
cmx_op_registry& cmx_op_registry::instance() {
    static cmx_op_registry registry;
    return registry;
}

// Register an operation
cmx_status cmx_op_registry::register_op(const std::string& name, const cmx_op& op) {
    if (op_count_ >= CMX_MAX_REGISTERED_OPS) {
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    // Check if already registered
    if (find_entry(name) != nullptr) {
        return cmx_status::ERROR_INVALID_ARGS; // Already exists
    }
    
    // Find free slot
    for (size_t i = 0; i < CMX_MAX_REGISTERED_OPS; ++i) {
        if (!ops_[i].used) {
            ops_[i].name = name;
            ops_[i].op = op;
            ops_[i].used = true;
            ++op_count_;
            return cmx_status::SUCCESS;
        }
    }
    
    return cmx_status::ERROR_OUT_OF_MEMORY;
}

// Get operation by name
const cmx_op* cmx_op_registry::get_op(const std::string& name) const {
    const op_entry* entry = find_entry(name);
    return entry ? &entry->op : nullptr;
}

// Check if operation is registered
bool cmx_op_registry::is_registered(const std::string& name) const {
    return find_entry(name) != nullptr;
}

// Get all registered operation names
void cmx_op_registry::get_registered_ops(std::string* names, size_t& count) const {
    size_t idx = 0;
    for (size_t i = 0; i < CMX_MAX_REGISTERED_OPS && idx < count; ++i) {
        if (ops_[i].used) {
            names[idx++] = ops_[i].name;
        }
    }
    count = idx;
}

// Clear all registrations
void cmx_op_registry::clear() {
    for (size_t i = 0; i < CMX_MAX_REGISTERED_OPS; ++i) {
        ops_[i].used = false;
        ops_[i].name.clear();
    }
    op_count_ = 0;
}

// Find entry by name (const version)
const cmx_op_registry::op_entry* cmx_op_registry::find_entry(const std::string& name) const {
    for (size_t i = 0; i < CMX_MAX_REGISTERED_OPS; ++i) {
        if (ops_[i].used && ops_[i].name == name) {
            return &ops_[i];
        }
    }
    return nullptr;
}

// Find entry by name (non-const version)
cmx_op_registry::op_entry* cmx_op_registry::find_entry(const std::string& name) {
    for (size_t i = 0; i < CMX_MAX_REGISTERED_OPS; ++i) {
        if (ops_[i].used && ops_[i].name == name) {
            return &ops_[i];
        }
    }
    return nullptr;
}

// C-style API functions
cmx_status cmx_register_op(const std::string& name, const cmx_op& op) {
    return cmx_op_registry::instance().register_op(name, op);
}

const cmx_op* cmx_get_op(const std::string& name) {
    return cmx_op_registry::instance().get_op(name);
}

bool cmx_is_op_registered(const std::string& name) {
    return cmx_op_registry::instance().is_registered(name);
}

void cmx_clear_op_registry() {
    cmx_op_registry::instance().clear();
}

size_t cmx_get_registered_op_count() {
    return cmx_op_registry::instance().size();
}

} // namespace cmx