#include <iostream>
#include <chrono>
#include <thread>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "cmx_config.hpp"

// Static variables for debouncing
static auto g_last_detection_time = std::chrono::steady_clock::now();
static int g_detection_count = 0;

// Get current timestamp as string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

// Check if enough time has passed since last detection (debouncing)
bool should_handle_detection() {
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - g_last_detection_time);
    
    if (time_since_last.count() >= DEBOUNCE_TIME_MS) {
        g_last_detection_time = current_time;
        return true;
    }
    
    return false;
}

// Simulate visual feedback (could be LED control in real hardware)
void trigger_visual_feedback() {
    std::cout << "💡 [VISUAL] Wake word LED ON" << std::endl;
    
    // Simulate brief LED flash
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "💡 [VISUAL] Wake word LED OFF" << std::endl;
}

// Simulate audio feedback (could be beep or sound in real hardware)
void trigger_audio_feedback() {
    std::cout << "🔊 [AUDIO] Wake word beep: BEEP!" << std::endl;
    
    // In real implementation, this might:
    // - Play a beep sound
    // - Send audio to speaker
    // - Trigger hardware buzzer
}

// Simulate UART/Serial output (for external device communication)
void trigger_uart_output() {
    std::cout << "📡 [UART] Sending wake signal to external device..." << std::endl;
    std::cout << "📡 [UART] TX: WAKE_DETECTED\\n" << std::endl;
    
    // In real implementation, this might:
    // - Send data over UART/Serial
    // - Communicate with another microcontroller
    // - Trigger external systems
}

// Log detection event (could be to file or database in real system)
void log_detection_event() {
    g_detection_count++;
    
    std::cout << "📝 [LOG] Wake detection #" << g_detection_count 
              << " at " << get_timestamp() << std::endl;
    
    // In real implementation, this might:
    // - Write to log file
    // - Send to remote logging service
    // - Store in local database
}

// Main wake word detection handler
void handle_wake_detected() {
    // Apply debouncing to prevent rapid repeated triggers
    if (!should_handle_detection()) {
        std::cout << "⏰ [DEBOUNCE] Wake detection ignored (too soon after last)" << std::endl;
        return;
    }
    
    // Print main detection message
    std::cout << std::endl;
    std::cout << "🎯 ===== WAKE WORD DETECTED! ===== 🎯" << std::endl;
    std::cout << "⏰ Time: " << get_timestamp() << std::endl;
    std::cout << "🔢 Detection count: " << g_detection_count + 1 << std::endl;
    std::cout << std::endl;
    
    // Execute various response actions
    try {
        // Log the event
        log_detection_event();
        
        // Trigger visual feedback
        trigger_visual_feedback();
        
        // Trigger audio feedback
        trigger_audio_feedback();
        
        // Send UART/Serial notification
        trigger_uart_output();
        
        // Additional custom actions can be added here:
        // - Start voice recording
        // - Activate voice assistant
        // - Send network notification
        // - Control smart home devices
        
        std::cout << "✅ [HANDLER] All wake word responses completed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ [ERROR] Exception in wake handler: " << e.what() << std::endl;
    }
    
    std::cout << "🎯 ===== WAKE HANDLING COMPLETE ===== 🎯" << std::endl;
    std::cout << std::endl;
}

// Optional: Function to reset detection statistics
void reset_wake_statistics() {
    g_detection_count = 0;
    g_last_detection_time = std::chrono::steady_clock::now();
    std::cout << "📊 [STATS] Wake detection statistics reset" << std::endl;
}

// Optional: Function to get detection statistics
int get_detection_count() {
    return g_detection_count;
}

// Optional: Test function to simulate wake detection (for debugging)
void test_wake_detection() {
    std::cout << "🧪 [TEST] Simulating wake word detection..." << std::endl;
    handle_wake_detected();
}