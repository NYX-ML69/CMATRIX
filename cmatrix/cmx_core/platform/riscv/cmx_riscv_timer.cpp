#include "cmx_riscv_timer.hpp"

namespace cmx::platform::riscv {

// Static member definitions
uint64_t Timer::cpu_frequency_hz_ = 100000000ULL; // Default 100MHz, should be set during init
Timer::cycles_t Timer::init_cycles_ = 0;
bool Timer::initialized_ = false;

bool Timer::initialize() {
    if (initialized_) {
        return true;
    }

    // Record initialization time
    init_cycles_ = get_cycles();
    initialized_ = true;
    
    return true;
}

void Timer::set_cpu_frequency(uint64_t freq_hz) {
    cpu_frequency_hz_ = freq_hz;
}

Timer::microseconds_t Timer::get_microseconds() {
    if (!initialized_) {
        return 0;
    }
    
    cycles_t current_cycles = get_cycles();
    cycles_t elapsed_cycles = current_cycles - init_cycles_;
    
    return cycles_to_microseconds(elapsed_cycles);
}

Timer::nanoseconds_t Timer::get_nanoseconds() {
    if (!initialized_) {
        return 0;
    }
    
    cycles_t current_cycles = get_cycles();
    cycles_t elapsed_cycles = current_cycles - init_cycles_;
    
    return cycles_to_nanoseconds(elapsed_cycles);
}

void Timer::delay_microseconds(microseconds_t microseconds) {
    if (microseconds == 0) {
        return;
    }
    
    cycles_t delay_cycles = (microseconds * cpu_frequency_hz_) / 1000000ULL;
    delay_cycles(delay_cycles);
}

void Timer::delay_nanoseconds(nanoseconds_t nanoseconds) {
    if (nanoseconds == 0) {
        return;
    }
    
    // For very small delays, ensure we delay at least 1 cycle
    cycles_t delay_cycles = (nanoseconds * cpu_frequency_hz_) / 1000000000ULL;
    if (delay_cycles == 0) {
        delay_cycles = 1;
    }
    
    delay_cycles(delay_cycles);
}

} // namespace cmx::platform::riscv