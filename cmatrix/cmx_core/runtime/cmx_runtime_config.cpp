// cmx_runtime_config.cpp
#include "cmx_runtime_config.hpp"

namespace cmx {
namespace runtime {

// Global runtime configuration instance
static RuntimeConfig g_config;

/**
 * @brief Get the global runtime configuration
 */
const RuntimeConfig& GetRuntimeConfig() {
    return g_config;
}

/**
 * @brief Initialize runtime configuration with custom values
 * @param config Custom configuration parameters
 */
void InitializeRuntimeConfig(const RuntimeConfig& config) {
    g_config = config;
}

} // namespace runtime
} // namespace cmx