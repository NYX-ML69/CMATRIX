// cmx_nios2_timer.cpp
// CMatrix Framework Implementation
#include "cmx_nios2_timer.hpp"

// Nios II HAL includes
#ifdef __NIOS2__
#include "system.h"
#include "alt_types.h"
#include "sys/alt_timestamp.h"
#include "sys/alt_alarm.h"
#include "io.h"
#else
// Fallback for non-Nios II compilation
#include <chrono>
#include <thread>
#endif

namespace cmx {
namespace platform {
namespace nios2 {

// Static variables for timer management
static bool timer_initialized = false;
static uint32_t timer_frequency_hz = TimerConfig::TIMER_FREQ_HZ;
static uint64_t timer_base_offset = 0;
static TimerStats timer_stats = {0};

#ifdef __NIOS2__
// Nios II specific timer implementation

void cmx_timer_init(uint32_t system_freq_hz) {
    if (timer_initialized) {
        return;
    }
    
    // Use provided frequency or detect from system
    if (system_freq_hz > 0) {
        timer_frequency_hz = system_freq_hz;
    } else {
        // Try to get frequency from HAL
        #ifdef ALT_CPU_FREQ
        timer_frequency_hz = ALT_CPU_FREQ;
        #else
        timer_frequency_hz = TimerConfig::TIMER_FREQ_HZ;
        #endif
    }
    
    // Initialize timestamp timer if available
    if (alt_timestamp_start() < 0) {
        // Fallback: use system timer or custom implementation
        timer_frequency_hz = ALT_CPU_FREQ;
    }
    
    // Record base time offset
    timer_base_offset = cmx_get_raw_timer();
    timer_initialized = true;
    
    // Reset statistics
    cmx_timer_reset_stats();
}

uint64_t cmx_get_raw_timer() {
    #ifdef ALT_TIMESTAMP_CLK_BASE
    return (uint64_t)alt_timestamp();
    #else
    // Fallback: read system timer directly
    static uint32_t high_word = 0;
    static uint32_t last_low = 0;
    
    uint32_t current_low = IORD_32DIRECT(0, 0); // Read timer register
    
    // Check for overflow
    if (current_low < last_low) {
        high_word++;
        timer_stats.timer_overflows++;
    }
    last_low = current_low;
    
    return ((uint64_t)high_word << 32) | current_low;
    #endif
}

uint64_t cmx_now_us() {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    uint64_t current_ticks = cmx_get_raw_timer() - timer_base_offset;
    return cmx_ticks_to_us(current_ticks);
}

void cmx_delay_us(uint32_t us) {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    timer_stats.total_delay_calls++;
    timer_stats.total_delay_time_us += us;
    
    if (us > timer_stats.max_delay_time_us) {
        timer_stats.max_delay_time_us = us;
    }
    
    uint64_t start = cmx_now_us();
    uint64_t target = start + us;
    
    // Busy wait for small delays
    if (us < 1000) {
        while (cmx_now_us() < target) {
            // Busy wait
        }
    } else {
        // Use HAL alarm for longer delays
        #ifdef ALT_ENHANCED_INTERRUPT_API_PRESENT
        alt_alarm alarm;
        volatile bool alarm_fired = false;
        
        auto alarm_callback = [](void* context) -> alt_u32 {
            *(volatile bool*)context = true;
            return 0; // Don't reschedule
        };
        
        alt_alarm_start(&alarm, us, alarm_callback, (void*)&alarm_fired);
        
        while (!alarm_fired) {
            // Yield CPU if possible
            usleep(1);
        }
        #else
        // Fallback busy wait
        while (cmx_now_us() < target) {
            // Busy wait
        }
        #endif
    }
}

#else
// Non-Nios II fallback implementation using standard C++

void cmx_timer_init(uint32_t system_freq_hz) {
    if (timer_initialized) {
        return;
    }
    
    timer_frequency_hz = system_freq_hz > 0 ? system_freq_hz : TimerConfig::TIMER_FREQ_HZ;
    timer_base_offset = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    timer_initialized = true;
    cmx_timer_reset_stats();
}

uint64_t cmx_get_raw_timer() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

uint64_t cmx_now_us() {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    return cmx_get_raw_timer() - timer_base_offset;
}

void cmx_delay_us(uint32_t us) {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    timer_stats.total_delay_calls++;
    timer_stats.total_delay_time_us += us;
    
    if (us > timer_stats.max_delay_time_us) {
        timer_stats.max_delay_time_us = us;
    }
    
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}

#endif

// Common implementations

uint64_t cmx_now_ms() {
    return cmx_now_us() / 1000;
}

void cmx_delay_ms(uint32_t ms) {
    if (ms == 0) return;
    
    // Break down large delays to avoid overflow
    while (ms > 1000) {
        cmx_delay_us(1000000); // 1 second
        ms -= 1000;
    }
    
    cmx_delay_us(ms * 1000);
}

bool cmx_delay_elapsed(uint64_t start_time_us, uint32_t duration_us) {
    uint64_t current = cmx_now_us();
    return (current - start_time_us) >= duration_us;
}

const TimerStats& cmx_timer_get_stats() {
    return timer_stats;
}

void cmx_timer_reset_stats() {
    timer_stats.total_delay_calls = 0;
    timer_stats.total_delay_time_us = 0;
    timer_stats.max_delay_time_us = 0;
    timer_stats.timer_overflows = 0;
}

uint64_t cmx_ticks_to_us(uint64_t ticks) {
    if (timer_frequency_hz == 0) {
        return 0;
    }
    
    // Avoid overflow in multiplication
    if (ticks > UINT64_MAX / TimerConfig::US_PER_SECOND) {
        return (ticks / timer_frequency_hz) * TimerConfig::US_PER_SECOND;
    } else {
        return (ticks * TimerConfig::US_PER_SECOND) / timer_frequency_hz;
    }
}

uint64_t cmx_us_to_ticks(uint64_t us) {
    if (timer_frequency_hz == 0) {
        return 0;
    }
    
    // Avoid overflow in multiplication
    if (us > UINT64_MAX / timer_frequency_hz) {
        return (us / TimerConfig::US_PER_SECOND) * timer_frequency_hz;
    } else {
        return (us * timer_frequency_hz) / TimerConfig::US_PER_SECOND;
    }
}

uint32_t cmx_timer_calibrate(uint32_t reference_delay_ms) {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    // Measure actual delay time
    uint64_t start_raw = cmx_get_raw_timer();
    cmx_delay_ms(reference_delay_ms);
    uint64_t end_raw = cmx_get_raw_timer();
    
    uint64_t elapsed_ticks = end_raw - start_raw;
    uint64_t expected_us = reference_delay_ms * 1000;
    
    if (elapsed_ticks == 0 || expected_us == 0) {
        return 0; // Calibration failed
    }
    
    // Calculate measured frequency
    uint32_t measured_freq = static_cast<uint32_t>(
        (elapsed_ticks * TimerConfig::US_PER_SECOND) / expected_us
    );
    
    // Update frequency if measurement seems reasonable (within 50% of expected)
    if (measured_freq > timer_frequency_hz / 2 && 
        measured_freq < timer_frequency_hz * 2) {
        timer_frequency_hz = measured_freq;
        return measured_freq;
    }
    
    return 0; // Calibration failed - frequency out of reasonable range
}

bool cmx_timer_self_test() {
    if (!timer_initialized) {
        cmx_timer_init();
    }
    
    // Test 1: Basic timer reading
    uint64_t time1 = cmx_now_us();
    uint64_t time2 = cmx_now_us();
    
    if (time2 < time1) {
        return false; // Timer going backwards
    }
    
    // Test 2: Short delay accuracy
    uint64_t delay_start = cmx_now_us();
    cmx_delay_us(1000); // 1ms delay
    uint64_t delay_end = cmx_now_us();
    
    uint64_t actual_delay = delay_end - delay_start;
    
    // Allow 50% tolerance for short delays
    if (actual_delay < 500 || actual_delay > 2000) {
        return false; // Delay too inaccurate
    }
    
    // Test 3: Timer frequency sanity check
    if (timer_frequency_hz < 1000 || timer_frequency_hz > 1000000000) {
        return false; // Unreasonable frequency
    }
    
    // Test 4: Raw timer advancing
    uint64_t raw1 = cmx_get_raw_timer();
    cmx_delay_us(10); // Very short delay
    uint64_t raw2 = cmx_get_raw_timer();
    
    if (raw2 <= raw1) {
        return false; // Raw timer not advancing
    }
    
    return true; // All tests passed
}

uint32_t cmx_timer_get_frequency() {
    return timer_frequency_hz;
}