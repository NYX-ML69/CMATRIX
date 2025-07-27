#include "cmx_cortex_m_timer.hpp"

#ifdef __arm__
#include "cmsis_gcc.h"
#endif

namespace cmx {
namespace platform {
namespace cortex_m {

// Static variables for timer management
static volatile uint32_t systick_millis = 0;
static volatile uint64_t systick_micros = 0;
static uint32_t systick_frequency = 0;
static uint32_t cycles_per_microsecond = 0;
static bool timer_initialized = false;

// Timer callback storage (simple implementation for embedded)
static constexpr uint8_t MAX_TIMERS = 4;
static struct {
    bool active;
    bool periodic;
    uint32_t timeout_ms;
    uint32_t remaining_ms;
    TimerCallback callback;
    uint32_t timer_id;
} timer_slots[MAX_TIMERS] = {};

/**
 * @brief SysTick interrupt handler
 */
extern "C" void SysTick_Handler(void) {
    systick_millis++;
    systick_micros += 1000;
    
    // Process software timers
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (timer_slots[i].active && timer_slots[i].remaining_ms > 0) {
            timer_slots[i].remaining_ms--;
            
            if (timer_slots[i].remaining_ms == 0) {
                // Timer expired, execute callback
                if (timer_slots[i].callback) {
                    timer_slots[i].callback(timer_slots[i].timer_id);
                }
                
                if (timer_slots[i].periodic) {
                    // Restart periodic timer
                    timer_slots[i].remaining_ms = timer_slots[i].timeout_ms;
                } else {
                    // One-shot timer, deactivate
                    timer_slots[i].active = false;
                }
            }
        }
    }
}

bool Timer::init(const TimerConfig& config) {
    if (timer_initialized) {
        return true;
    }
    
    // Initialize timer slots
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        timer_slots[i].active = false;
    }
    
    systick_frequency = config.frequency_hz;
    
#ifdef __arm__
    // Configure SysTick for 1ms interrupts
    uint32_t system_clock = SystemCoreClock;
    cycles_per_microsecond = system_clock / 1000000;
    
    // Configure SysTick to interrupt every 1ms
    if (SysTick_Config(system_clock / 1000) != 0) {
        return false;
    }
    
    // Set SysTick interrupt priority if interrupts enabled
    if (config.enable_interrupts) {
        NVIC_SetPriority(SysTick_IRQn, config.interrupt_priority);
    }
#endif
    
    timer_initialized = true;
    return true;
}

void Timer::deinit() {
    if (!timer_initialized) {
        return;
    }
    
#ifdef __arm__
    // Disable SysTick
    SysTick->CTRL = 0;
#endif
    
    // Clear all timers
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        timer_slots[i].active = false;
    }
    
    timer_initialized = false;
}

uint32_t Timer::get_millis() {
    if (!timer_initialized) {
        return 0;
    }
    return systick_millis;
}

uint64_t Timer::get_micros() {
    if (!timer_initialized) {
        return 0;
    }
    
#ifdef __arm__
    // Get high-precision microsecond count using SysTick counter
    __disable_irq();
    uint32_t millis = systick_millis;
    uint32_t ticks = SysTick->VAL;
    uint32_t reload = SysTick->LOAD;
    __enable_irq();
    
    // Calculate microseconds within current millisecond
    uint32_t us_in_ms = ((reload - ticks) * 1000) / reload;
    return (static_cast<uint64_t>(millis) * 1000) + us_in_ms;
#else
    return systick_micros;
#endif
}

void Timer::delay_ms(uint32_t ms) {
    if (!timer_initialized || ms == 0) {
        return;
    }
    
    uint32_t start_time = get_millis();
    while ((get_millis() - start_time) < ms) {
        // Non-blocking delay using timer
        #ifdef __arm__
        __WFI(); // Wait for interrupt to save power
        #endif
    }
}

void Timer::delay_us(uint32_t us) {
    if (!timer_initialized || us == 0) {
        return;
    }
    
#ifdef __arm__
    // Use DWT cycle counter for precise microsecond delays if available
    if (DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) {
        uint32_t start_cycles = DWT->CYCCNT;
        uint32_t cycles_to_wait = us * cycles_per_microsecond;
        
        while ((DWT->CYCCNT - start_cycles) < cycles_to_wait) {
            // Busy wait for precise timing
        }
        return;
    }
#endif
    
    // Fallback: use millisecond timer with busy loop
    uint64_t start_time = get_micros();
    while ((get_micros() - start_time) < us) {
        // Busy wait
    }
}

bool Timer::start_oneshot(uint32_t timeout_ms, TimerCallback callback, uint32_t timer_id) {
    if (!timer_initialized || !callback || timeout_ms == 0) {
        return false;
    }
    
    // Find available timer slot
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (!timer_slots[i].active) {
            timer_slots[i].active = true;
            timer_slots[i].periodic = false;
            timer_slots[i].timeout_ms = timeout_ms;
            timer_slots[i].remaining_ms = timeout_ms;
            timer_slots[i].callback = callback;
            timer_slots[i].timer_id = timer_id;
            return true;
        }
    }
    
    return false; // No available slots
}

bool Timer::start_periodic(uint32_t period_ms, TimerCallback callback, uint32_t timer_id) {
    if (!timer_initialized || !callback || period_ms == 0) {
        return false;
    }
    
    // Find available timer slot
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (!timer_slots[i].active) {
            timer_slots[i].active = true;
            timer_slots[i].periodic = true;
            timer_slots[i].timeout_ms = period_ms;
            timer_slots[i].remaining_ms = period_ms;
            timer_slots[i].callback = callback;
            timer_slots[i].timer_id = timer_id;
            return true;
        }
    }
    
    return false; // No available slots
}

void Timer::stop_timer(uint32_t timer_id) {
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (timer_slots[i].active && timer_slots[i].timer_id == timer_id) {
            timer_slots[i].active = false;
            break;
        }
    }
}

void Timer::reset_counters() {
    if (!timer_initialized) {
        return;
    }
    
    __disable_irq();
    systick_millis = 0;
    systick_micros = 0;
    __enable_irq();
}

uint32_t Timer::get_resolution_us() {
    if (!timer_initialized) {
        return 1000; // 1ms default
    }
    
#ifdef __arm__
    // If DWT is available, we have cycle-accurate timing
    if (DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk) {
        return 1; // 1μs resolution
    }
#endif
    
    return 1000; // 1ms resolution via SysTick
}

bool Timer::is_initialized() {
    return timer_initialized;
}

// ProfileTimer implementation
ProfileTimer::ProfileTimer() {
    start_time_us_ = Timer::get_micros();
}

uint64_t ProfileTimer::elapsed_us() const {
    return Timer::get_micros() - start_time_us_;
}

uint32_t ProfileTimer::elapsed_ms() const {
    return static_cast<uint32_t>(elapsed_us() / 1000);
}

void ProfileTimer::reset() {
    start_time_us_ = Timer::get_micros();
}

} // namespace cortex_m
} // namespace platform
} // namespace cmx