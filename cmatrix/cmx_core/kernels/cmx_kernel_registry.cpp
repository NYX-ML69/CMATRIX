#include "cmx_kernel_registry.hpp"

#ifdef ARM_MATH_CM4
#include "arm_math.h"
#endif

namespace cmx {
namespace kernels {

CmxKernelRegistry& CmxKernelRegistry::instance() {
    static CmxKernelRegistry instance_;
    return instance_;
}

bool CmxKernelRegistry::register_kernel(
    KernelType type,
    KernelFactory factory,
    uint32_t capabilities
) {
    if (!factory) {
        return false;
    }
    
    type_registry_.emplace(type, KernelEntry(factory, capabilities));
    return true;
}

bool CmxKernelRegistry::register_kernel(
    const std::string& name,
    KernelFactory factory,
    uint32_t capabilities
) {
    if (!factory || name.empty()) {
        return false;
    }
    
    name_registry_.emplace(name, KernelEntry(factory, capabilities));
    return true;
}

std::unique_ptr<CmxKernel> CmxKernelRegistry::create_kernel(KernelType type) {
    auto it = type_registry_.find(type);
    if (it != type_registry_.end() && capabilities_satisfied(it->second.required_capabilities)) {
        return it->second.factory();
    }
    return nullptr;
}

std::unique_ptr<CmxKernel> CmxKernelRegistry::create_kernel(const std::string& name) {
    auto it = name_registry_.find(name);
    if (it != name_registry_.end() && capabilities_satisfied(it->second.required_capabilities)) {
        return it->second.factory();
    }
    return nullptr;
}

bool CmxKernelRegistry::is_supported(KernelType type) const {
    auto it = type_registry_.find(type);
    return it != type_registry_.end() && capabilities_satisfied(it->second.required_capabilities);
}

bool CmxKernelRegistry::is_supported(const std::string& name) const {
    auto it = name_registry_.find(name);
    return it != name_registry_.end() && capabilities_satisfied(it->second.required_capabilities);
}

std::vector<KernelType> CmxKernelRegistry::get_supported_types() const {
    std::vector<KernelType> types;
    for (const auto& pair : type_registry_) {
        if (capabilities_satisfied(pair.second.required_capabilities)) {
            types.push_back(pair.first);
        }
    }
    return types;
}

std::vector<std::string> CmxKernelRegistry::get_supported_names() const {
    std::vector<std::string> names;
    for (const auto& pair : name_registry_) {
        if (capabilities_satisfied(pair.second.required_capabilities)) {
            names.push_back(pair.first);
        }
    }
    return names;
}

void CmxKernelRegistry::set_hardware_capabilities(uint32_t capabilities) {
    hardware_capabilities_ = capabilities;
}

uint32_t CmxKernelRegistry::get_hardware_capabilities() const {
    return hardware_capabilities_;
}

const char* CmxKernelRegistry::type_to_string(KernelType type) {
    switch (type) {
        case KernelType::CONV2D: return "conv2d";
        case KernelType::CONV3D: return "conv3d";
        case KernelType::DEPTHWISE_CONV: return "depthwise_conv";
        case KernelType::DENSE: return "dense";
        case KernelType::BIAS: return "bias";
        case KernelType::POOLING: return "pooling";
        case KernelType::MATMUL: return "matmul";
        case KernelType::NORMALIZATION: return "normalization";
        case KernelType::LSTM: return "lstm";
        case KernelType::GRU: return "gru";
        case KernelType::RNN: return "rnn";
        default: return "unknown";
    }
}

KernelType CmxKernelRegistry::string_to_type(const std::string& name) {
    if (name == "conv2d") return KernelType::CONV2D;
    if (name == "conv3d") return KernelType::CONV3D;
    if (name == "depthwise_conv") return KernelType::DEPTHWISE_CONV;
    if (name == "dense") return KernelType::DENSE;
    if (name == "bias") return KernelType::BIAS;
    if (name == "pooling") return KernelType::POOLING;
    if (name == "matmul") return KernelType::MATMUL;
    if (name == "normalization") return KernelType::NORMALIZATION;
    if (name == "lstm") return KernelType::LSTM;
    if (name == "gru") return KernelType::GRU;
    if (name == "rnn") return KernelType::RNN;
    return KernelType::UNKNOWN;
}

void CmxKernelRegistry::initialize() {
    // Detect hardware capabilities
    hardware_capabilities_ = detect_hardware_capabilities();
    
    // Note: Kernel registration is handled by AutoKernelRegister
    // instances created at static initialization time
}

uint32_t CmxKernelRegistry::detect_hardware_capabilities() {
    uint32_t capabilities = 0;
    
    #ifdef ARM_MATH_CM4
    capabilities |= static_cast<uint32_t>(HardwareCapability::CMSIS_DSP);
    #endif
    
    #ifdef __ARM_NEON
    capabilities |= static_cast<uint32_t>(HardwareCapability::NEON);
    #endif
    
    #ifdef __ARM_FP
    capabilities |= static_cast<uint32_t>(HardwareCapability::FPU);
    #endif
    
    #ifdef __XTENSA__
    capabilities |= static_cast<uint32_t>(HardwareCapability::XTENSA);
    #endif
    
    // Check for SIMD support
    #if defined(__ARM_NEON) || defined(__XTENSA__)
    capabilities |= static_cast<uint32_t>(HardwareCapability::SIMD);
    #endif
    
    return capabilities;
}

bool CmxKernelRegistry::capabilities_satisfied(uint32_t required) const {
    return (hardware_capabilities_ & required) == required;
}

} // namespace kernels
} // namespace cmx