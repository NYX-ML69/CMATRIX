// cmx_nios2_port.cpp
// CMatrix Framework Implementation
/**
 * @file cmx_nios2_port.cpp
 * @brief Core platform abstraction layer implementation for Nios II
 * @author CMatrix Development Team
 * @version 1.0
 */

#include "cmx_nios2_port.hpp"

#include <system.h>
#include <sys/alt_cache.h>
#include <sys/alt_irq.h>
#include <sys/alt_timestamp.h>
#include <altera_avalon_timer_regs.h>
#include <altera_avalon_pio_regs.h>
#include <io.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

namespace cmx {
namespace platform {
namespace nios2 {

// =============================================================================
// Static Variables
// =============================================================================

static PlatformCapabilities g_capabilities;
static SystemStats g_stats = {0};
static SystemStatus g_system_status = SystemStatus::UNINITIALIZED;
static uint64_t g_system_start_time = 0;
static uint64_t g_profile_starts[CMX_MAX_PROFILE_SESSIONS] = {0};

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Initialize platform capabilities structure
 */
static void init_platform_capabilities() {
    g_capabilities = {};
    
    // CPU frequency
#ifdef ALT_CPU_FREQ
    g_capabilities.cpu_frequency_hz = ALT_CPU_FREQ;
#else
    g_capabilities.cpu_frequency_hz = 50000000; // Default 50MHz
#endif

    // Memory size estimation
#ifdef ONCHIP_MEMORY2_0_SPAN
    g_capabilities.memory_size = ONCHIP_MEMORY2_0_SPAN;
#else
    g_capabilities.memory_size = 65536; // Default 64KB
#endif

    // Cache support
#ifdef ALT_CPU_DCACHE_SIZE
    g_capabilities.supported_features |= static_cast<uint32_t>(PlatformFeature::CACHE_SUPPORT);
    g_capabilities.cache_line_size = ALT_CPU_DCACHE_LINE_SIZE;
#else
    g_capabilities.cache_line_size = 0;
#endif

    // Timer support
#ifdef SYS_CLK_TIMER_BASE
    g_capabilities.supported_features |= static_cast<uint32_t>(PlatformFeature::TIMER_SUPPORT);
    g_capabilities.timer_resolution_us = CMX_DEFAULT_TIMER_RESOLUTION_US;
#endif

    // Interrupt support
    g_capabilities.supported_features |= static_cast<uint32_t>(PlatformFeature::INTERRUPT_SUPPORT);
    g_capabilities.interrupt_levels = 32; // Nios II supports up to 32 interrupts

    // DMA channels (platform dependent)
#ifdef SGDMA_0_BASE
    g_capabilities.supported_features |= static_cast<uint32_t>(PlatformFeature::DMA_SUPPORT);
    g_capabilities.dma_channels = 1;
#endif

    // Performance counters (if available)
#ifdef PERFORMANCE_COUNTER_BASE
    g_capabilities.supported_features |= static_cast<uint32_t>(PlatformFeature::PERFORMANCE_COUNTERS);
#endif
}

/**
 * @brief Format log message with timestamp
 */
static void format_log_message(char* buffer, size_t buffer_size, LogLevel level, const char* message) {
    const char* level_str;
    switch (level) {
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::INFO:    level_str = "INFO";  break;
        case LogLevel::WARNING: level_str = "WARN";  break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
        case LogLevel::FATAL:   level_str = "FATAL"; break;
        default:                level_str = "UNKNOWN"; break;
    }
    
    uint64_t timestamp = cmx_get_timestamp_us();
    uint32_t seconds = static_cast<uint32_t>(timestamp / 1000000);
    uint32_t microseconds = static_cast<uint32_t>(timestamp % 1000000);
    
    snprintf(buffer, buffer_size, "[%lu.%06lu] %s: %s", 
             (unsigned long)seconds, (unsigned long)microseconds, level_str, message);
}

// =============================================================================
// Core Platform Functions
// =============================================================================

bool cmx_platform_init() {
    if (g_system_status != SystemStatus::UNINITIALIZED) {
        return true; // Already initialized
    }
    
    g_system_status = SystemStatus::INITIALIZING;
    
    // Initialize platform capabilities
    init_platform_capabilities();
    
    // Initialize timestamp system
    if (alt_timestamp_start() < 0) {
        g_system_status = SystemStatus::ERROR;
        cmx_set_error_code(0x0001); // Timestamp initialization failed
        return false;
    }
    
    // Record system start time
    g_system_start_time = cmx_get_timestamp_us();
    
    // Initialize statistics
    memset(&g_stats, 0, sizeof(g_stats));
    
    g_system_status = SystemStatus::READY;
    return true;
}

void cmx_platform_cleanup() {
    if (g_system_status == SystemStatus::UNINITIALIZED) {
        return;
    }
    
    g_system_status = SystemStatus::SHUTDOWN;
    
    // Cleanup any platform-specific resources
    // (Most Nios II resources are managed by HAL)
}

uint64_t cmx_get_timestamp_us() {
    if (g_capabilities.supported_features & static_cast<uint32_t>(PlatformFeature::TIMER_SUPPORT)) {
        // Use high-resolution timestamp if available
        alt_timestamp_type timestamp = alt_timestamp();
        
        // Convert timestamp to microseconds
        uint64_t freq = alt_timestamp_freq();
        if (freq > 0) {
            return (timestamp * 1000000ULL) / freq;
        }
    }
    
    // Fallback: estimate based on CPU cycles
    // This is less accurate but works on all systems
    static uint64_t cycle_counter = 0;
    cycle_counter += g_capabilities.cpu_frequency_hz / 1000000; // Approximate
    return cycle_counter;
}

void cmx_log(LogLevel level, const char* message) {
    char formatted_message[CMX_MAX_LOG_MESSAGE_SIZE + 64];
    format_log_message(formatted_message, sizeof(formatted_message), level, message);
    
    // Output to UART/JTAG console
    printf("%s\n", formatted_message);
    fflush(stdout);
    
    // Update statistics
    if (level >= LogLevel::ERROR) {
        g_stats.last_error_code = static_cast<uint16_t>(level);
    }
}

void cmx_log(const char* message) {
    cmx_log(LogLevel::INFO, message);
}

void cmx_yield() {
    // On Nios II, yielding is typically done by enabling interrupts briefly
    // or using a small delay to allow interrupt processing
    usleep(1); // 1 microsecond delay
}

void cmx_system_reset() {
    cmx_log(LogLevel::INFO, "System reset requested");
    
    // Perform soft reset using Nios II reset mechanism
    // This is platform-dependent and may require custom reset logic
    
#ifdef RESET_REQUEST_PIO_BASE
    // Trigger reset via PIO if available
    IOWR_ALTERA_AVALON_PIO_DATA(RESET_REQUEST_PIO_BASE, 1);
#else
    // Software-based reset (less reliable)
    while (true) {
        // Infinite loop - watchdog should reset system
        cmx_yield();
    }
#endif
}

// =============================================================================
// Platform Information Functions
// =============================================================================

const PlatformCapabilities& cmx_get_platform_capabilities() {
    return g_capabilities;
}

SystemStatus cmx_get_system_status() {
    return g_system_status;
}

const SystemStats& cmx_get_system_stats() {
    // Update dynamic statistics
    g_stats.uptime_us = cmx_get_timestamp_us() - g_system_start_time;
    g_stats.interrupt_count++; // Approximate - actual count would need ISR hooks
    
    return g_stats;
}

bool cmx_is_feature_supported(PlatformFeature feature) {
    return (g_capabilities.supported_features & static_cast<uint32_t>(feature)) != 0;
}

uint32_t cmx_get_cpu_frequency() {
    return g_capabilities.cpu_frequency_hz;
}

uint32_t cmx_get_available_memory() {
    return g_capabilities.memory_size;
}

// =============================================================================
// Cache Management Functions
// =============================================================================

void cmx_cache_flush_range(void* addr, size_t size) {
#ifdef ALT_CPU_DCACHE_SIZE
    alt_dcache_flush(addr, size);
#else
    // No cache support - function is no-op
    (void)addr;
    (void)size;
#endif
}

void cmx_cache_invalidate_range(void* addr, size_t size) {
#ifdef ALT_CPU_DCACHE_SIZE
    alt_dcache_flush(addr, size); // Nios II HAL uses flush for invalidate
#else
    // No cache support - function is no-op
    (void)addr;
    (void)size;
#endif
}

void cmx_cache_flush_all() {
#ifdef ALT_CPU_DCACHE_SIZE
    alt_dcache_flush_all();
#endif
}

void cmx_cache_invalidate_all() {
#ifdef ALT_CPU_DCACHE_SIZE
    alt_dcache_flush_all(); // Nios II HAL uses flush for invalidate
#endif
}

// =============================================================================
// Interrupt Management Functions
// =============================================================================

uint32_t cmx_disable_interrupts() {
    return alt_irq_disable_all();
}

void cmx_restore_interrupts(uint32_t state) {
    alt_irq_enable_all(state);
}

bool cmx_enable_interrupt(uint32_t irq_num) {
    return alt_ic_irq_enable(ALT_IRQ_CPU, irq_num) == 0;
}

bool cmx_disable_interrupt(uint32_t irq_num) {
    return alt_ic_irq_disable(ALT_IRQ_CPU, irq_num) == 0;
}

// =============================================================================
// Error Handling Functions
// =============================================================================

void cmx_set_error_code(uint16_t error_code) {
    g_stats.last_error_code = error_code;
    if (error_code != 0) {
        g_system_status = SystemStatus::ERROR;
    }
}

uint16_t cmx_get_last_error() {
    return g_stats.last_error_code;
}

void cmx_clear_error() {
    g_stats.last_error_code = 0;
    if (g_system_status == SystemStatus::ERROR) {
        g_system_status = SystemStatus::READY;
    }
}

bool cmx_has_error() {
    return g_stats.last_error_code != 0;
}

// =============================================================================
// Power Management Functions
// =============================================================================

void cmx_enter_low_power(bool wake_on_interrupt) {
    if (wake_on_interrupt) {
        // Enable interrupts and wait
        uint32_t int_state = cmx_disable_interrupts();
        cmx_restore_interrupts(int_state);
        
        // Use CPU sleep instruction if available
        usleep(1000); // 1ms sleep as fallback
    } else {
        // Deep sleep without interrupt wake-up
        // This is platform-specific and may not be available
        usleep(10000); // 10ms sleep as fallback
    }
}

uint32_t cmx_get_power_consumption_mw() {
    // Power consumption estimation for Nios II
    // This is highly platform-dependent and typically requires external power monitoring
    
    // Rough estimation based on CPU frequency and activity
    uint32_t base_power = 50; // Base power in mW
    uint32_t freq_mhz = g_capabilities.cpu_frequency_hz / 1000000;
    
    // Simple linear model: higher frequency = higher power
    return base_power + (freq_mhz / 10);
}

// =============================================================================
// Debug and Profiling Functions
// =============================================================================

void cmx_profile_start(uint32_t profile_id) {
    if (profile_id < CMX_MAX_PROFILE_SESSIONS) {
        g_profile_starts[profile_id] = cmx_get_timestamp_us();
    }
}

uint64_t cmx_profile_end(uint32_t profile_id) {
    if (profile_id < CMX_MAX_PROFILE_SESSIONS) {
        uint64_t end_time = cmx_get_timestamp_us();
        uint64_t start_time = g_profile_starts[profile_id];
        g_profile_starts[profile_id] = 0;
        
        if (start_time > 0) {
            return end_time - start_time;
        }
    }
    return 0;
}

uint64_t cmx_get_cpu_cycles() {
    if (cmx_is_feature_supported(PlatformFeature::PERFORMANCE_COUNTERS)) {
#ifdef PERFORMANCE_COUNTER_BASE
        // Read cycle counter from performance counter peripheral
        return IORD_32DIRECT(PERFORMANCE_COUNTER_BASE, 0);
#endif
    }
    
    // Fallback: estimate cycles from timestamp
    uint64_t timestamp_us = cmx_get_timestamp_us();
    return cmx_us_to_cycles(timestamp_us);
}

uint64_t cmx_cycles_to_us(uint64_t cycles) {
    if (g_capabilities.cpu_frequency_hz == 0) {
        return 0;
    }
    return (cycles * 1000000ULL) / g_capabilities.cpu_frequency_hz;
}

uint64_t cmx_us_to_cycles(uint64_t us) {
    return (us * g_capabilities.cpu_frequency_hz) / 1000000ULL;
}

// =============================================================================
// Memory Barrier Functions
// =============================================================================

void cmx_memory_barrier() {
    // Nios II memory barrier - ensure all memory operations complete
    __asm__ volatile ("" ::: "memory");
    
    // If cache is present, flush to ensure coherency
    if (cmx_is_feature_supported(PlatformFeature::CACHE_SUPPORT)) {
        cmx_cache_flush_all();
    }
}

void cmx_data_memory_barrier() {
    // Data memory barrier for Nios II
    __asm__ volatile ("" ::: "memory");
}

void cmx_instruction_barrier() {
    // Instruction synchronization barrier
    __asm__ volatile (
        "flushp\n"  // Flush pipeline
        ::: "memory"
    );
}

} // namespace nios2
} // namespace platform
} // namespace cmx