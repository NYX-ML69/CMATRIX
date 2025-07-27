#pragma once

#include "cmx_kernel_interface.hpp"
#include <unordered_map>
#include <string>
#include <memory>

namespace cmx {
namespace kernels {

/**
 * @brief Kernel operation type enumeration
 */
enum class KernelType {
    CONV2D,
    CONV3D,
    DEPTHWISE_CONV,
    DENSE,
    BIAS,
    POOLING,
    MATMUL,
    NORMALIZATION,
    LSTM,
    GRU,
    RNN,
    UNKNOWN
};

/**
 * @brief Hardware capability flags
 */
enum class HardwareCapability {
    NONE = 0,
    NEON = 1 << 0,
    XTENSA = 1 << 1,
    CMSIS_DSP = 1 << 2,
    FPU = 1 << 3,
    SIMD = 1 << 4
};

/**
 * @brief Central registry for kernel implementations
 * 
 * This singleton class manages the registration and lookup of kernel
 * implementations. It supports hardware-specific kernel selection and
 * provides a factory pattern for kernel instantiation.
 */
class CmxKernelRegistry {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the registry instance
     */
    static CmxKernelRegistry& instance();

    /**
     * @brief Register a kernel factory function
     * @param type Kernel type identifier
     * @param factory Factory function for creating kernel instances
     * @param capabilities Hardware capabilities required
     * @return True if registration successful, false otherwise
     */
    bool register_kernel(
        KernelType type,
        KernelFactory factory,
        uint32_t capabilities = 0
    );

    /**
     * @brief Register a kernel factory function by string name
     * @param name String name of the kernel type
     * @param factory Factory function for creating kernel instances
     * @param capabilities Hardware capabilities required
     * @return True if registration successful, false otherwise
     */
    bool register_kernel(
        const std::string& name,
        KernelFactory factory,
        uint32_t capabilities = 0
    );

    /**
     * @brief Create a kernel instance by type
     * @param type Kernel type identifier
     * @return Unique pointer to kernel instance, nullptr if not found
     */
    std::unique_ptr<CmxKernel> create_kernel(KernelType type);

    /**
     * @brief Create a kernel instance by string name
     * @param name String name of the kernel type
     * @return Unique pointer to kernel instance, nullptr if not found
     */
    std::unique_ptr<CmxKernel> create_kernel(const std::string& name);

    /**
     * @brief Check if a kernel type is supported
     * @param type Kernel type identifier
     * @return True if supported, false otherwise
     */
    bool is_supported(KernelType type) const;

    /**
     * @brief Check if a kernel type is supported by string name
     * @param name String name of the kernel type
     * @return True if supported, false otherwise
     */
    bool is_supported(const std::string& name) const;

    /**
     * @brief Get list of all registered kernel types
     * @return Vector of registered kernel types
     */
    std::vector<KernelType> get_supported_types() const;

    /**
     * @brief Get list of all registered kernel names
     * @return Vector of registered kernel names
     */
    std::vector<std::string> get_supported_names() const;

    /**
     * @brief Set hardware capabilities for kernel selection
     * @param capabilities Bitmask of hardware capabilities
     */
    void set_hardware_capabilities(uint32_t capabilities);

    /**
     * @brief Get current hardware capabilities
     * @return Bitmask of hardware capabilities
     */
    uint32_t get_hardware_capabilities() const;

    /**
     * @brief Convert kernel type enum to string
     * @param type Kernel type enum
     * @return String representation
     */
    static const char* type_to_string(KernelType type);

    /**
     * @brief Convert string to kernel type enum
     * @param name String name
     * @return Kernel type enum
     */
    static KernelType string_to_type(const std::string& name);

    /**
     * @brief Initialize registry with default kernels
     * 
     * This method registers all available kernel implementations
     * and detects hardware capabilities automatically.
     */
    void initialize();

private:
    struct KernelEntry {
        KernelFactory factory;
        uint32_t required_capabilities;
        
        KernelEntry(KernelFactory f, uint32_t caps) 
            : factory(f), required_capabilities(caps) {}
    };

    CmxKernelRegistry() = default;
    ~CmxKernelRegistry() = default;
    
    // Disable copy and move
    CmxKernelRegistry(const CmxKernelRegistry&) = delete;
    CmxKernelRegistry& operator=(const CmxKernelRegistry&) = delete;

    /**
     * @brief Detect hardware capabilities at runtime
     * @return Bitmask of detected capabilities
     */
    uint32_t detect_hardware_capabilities();

    /**
     * @brief Check if hardware capabilities are satisfied
     * @param required Required capabilities
     * @return True if satisfied, false otherwise
     */
    bool capabilities_satisfied(uint32_t required) const;

    std::unordered_map<KernelType, KernelEntry> type_registry_;
    std::unordered_map<std::string, KernelEntry> name_registry_;
    uint32_t hardware_capabilities_ = 0;
};

/**
 * @brief Automatic kernel registration helper
 * 
 * This class provides RAII-style registration of kernels at static
 * initialization time.
 */
template<typename KernelClass>
class AutoKernelRegister {
public:
    AutoKernelRegister(KernelType type, uint32_t capabilities = 0) {
        CmxKernelRegistry::instance().register_kernel(
            type,
            []() -> std::unique_ptr<CmxKernel> {
                return std::make_unique<KernelClass>();
            },
            capabilities
        );
    }
    
    AutoKernelRegister(const std::string& name, uint32_t capabilities = 0) {
        CmxKernelRegistry::instance().register_kernel(
            name,
            []() -> std::unique_ptr<CmxKernel> {
                return std::make_unique<KernelClass>();
            },
            capabilities
        );
    }
};

/**
 * @brief Macro for automatic kernel registration
 */
#define REGISTER_KERNEL(KernelClass, Type, Capabilities) \
    static AutoKernelRegister<KernelClass> \
    g_##KernelClass##_register(Type, Capabilities)

} // namespace kernels
} // namespace cmx