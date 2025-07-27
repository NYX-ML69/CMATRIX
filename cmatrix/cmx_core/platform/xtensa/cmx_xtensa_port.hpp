#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx::platform::xtensa {

/**
 * @brief Initialize platform-specific systems
 */
void cmx_platform_init();

/**
 * @brief Get high-precision timestamp in microseconds
 * @return Current timestamp in microseconds since system start
 */
uint64_t cmx_get_timestamp_us();

/**
 * @brief Platform logging function
 * @param message Null-terminated string to log
 */
void cmx_log(const char* message);

/**
 * @brief Yield CPU control to other tasks/interrupts
 */
void cmx_yield();

/**
 * @brief Get CPU frequency in Hz
 * @return CPU frequency
 */
uint32_t cmx_get_cpu_freq_hz();

/**
 * @brief Critical section management
 */
class CriticalSection {
public:
    CriticalSection();
    ~CriticalSection();
    
private:
    uint32_t saved_int_level_;
};

/**
 * @brief Memory barrier functions for cache management
 */
void cmx_memory_barrier();
void cmx_data_cache_flush(void* addr, size_t size);
void cmx_instruction_cache_invalidate(void* addr, size_t size);

/**
 * @brief Platform-specific reset function
 */
void cmx_platform_reset();

/**
 * @brief Get available stack space
 * @return Bytes of available stack space
 */
size_t cmx_get_free_stack_size();

/**
 * @brief Platform info structure
 */
struct PlatformInfo {
    const char* platform_name;
    const char* cpu_model;
    uint32_t cpu_freq_hz;
    uint32_t total_memory_bytes;
    uint32_t available_memory_bytes;
    bool has_fpu;
    bool has_dma;
    uint8_t num_cores;
};

/**
 * @brief Get platform information
 * @return Platform info structure
 */
const PlatformInfo& cmx_get_platform_info();

} // namespace cmx::platform::xtensa

// Convenience macros
#define CMX_CRITICAL_SECTION() cmx::platform::xtensa::CriticalSection _cs
#define CMX_LOG(msg) cmx::platform::xtensa::cmx_log(msg)