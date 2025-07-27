#ifndef CMX_OP_REGISTRY_HPP
#define CMX_OP_REGISTRY_HPP

#include "cmx_ops.hpp"
#include <unordered_map>
#include <string>
#include <cstdint>

namespace cmx {

// Maximum number of registered operations (for static allocation)
constexpr size_t CMX_MAX_REGISTERED_OPS = 256;

// Operation registry class
class cmx_op_registry {
public:
    // Get singleton instance
    static cmx_op_registry& instance();
    
    // Register an operation
    cmx_status register_op(const std::string& name, const cmx_op& op);
    
    // Get operation by name
    const cmx_op* get_op(const std::string& name) const;
    
    // Check if operation is registered
    bool is_registered(const std::string& name) const;
    
    // Get all registered operation names
    void get_registered_ops(std::string* names, size_t& count) const;
    
    // Clear all registrations
    void clear();
    
    // Get registration count
    size_t size() const { return op_count_; }
    
private:
    cmx_op_registry() = default;
    ~cmx_op_registry() = default;
    
    // Static storage for operations (avoiding dynamic allocation)
    struct op_entry {
        std::string name;
        cmx_op op;
        bool used;
    };
    
    op_entry ops_[CMX_MAX_REGISTERED_OPS];
    size_t op_count_ = 0;
    
    // Find entry by name
    const op_entry* find_entry(const std::string& name) const;
    op_entry* find_entry(const std::string& name);
};

// C-style API functions
cmx_status cmx_register_op(const std::string& name, const cmx_op& op);
const cmx_op* cmx_get_op(const std::string& name);
bool cmx_is_op_registered(const std::string& name);
void cmx_clear_op_registry();
size_t cmx_get_registered_op_count();

// Static registration helper macro
#define CMX_REGISTER_OP_STATIC(name, func, inputs, outputs, attrs, inplace) \
    static bool _cmx_reg_##name = []() { \
        cmx_op op = {#name, func, inputs, outputs, attrs, inplace, 1}; \
        return cmx_register_op(#name, op) == cmx_status::SUCCESS; \
    }();

// Inline registration for link-time optimization
template<typename F>
constexpr inline bool cmx_register_op_inline(const char* name, F func, 
                                             uint32_t inputs, uint32_t outputs,
                                             uint32_t attrs = 0, bool inplace = false) {
    cmx_op op = {name, func, inputs, outputs, attrs, inplace, 1};
    return cmx_register_op(name, op) == cmx_status::SUCCESS;
}

} // namespace cmx

#endif // CMX_OP_REGISTRY_HPP