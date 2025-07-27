#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstring>
#include <random>

#include "cmx_config.hpp"

// Platform-specific includes would go here
// For this demo, we'll use simulated audio input

// Audio buffer and synchronization
static std::queue<int16_t> g_audio_buffer;
static std::mutex g_buffer_mutex;
static bool g_mic_initialized = false;
static std::thread g_audio_thread;
static bool g_audio_thread_running = false;

// Simulated audio generation (replace with real microphone interface)
static std::random_device g_rd;
static std::mt19937 g_gen(g_rd());
static std::uniform_int_distribution<int16_t> g_noise_dist(-100, 100);

// Generate simulated audio data
void generate_simulated_audio(int16_t* buffer, size_t samples) {
    static float phase = 0.0f;
    static int wake_word_counter = 0;
    
    for (size_t i = 0; i < samples; i++) {
        int16_t sample = 0;
        
        // Add background noise
        sample += g_noise_dist(g_gen);
        
        // Occasionally simulate a wake word pattern (higher amplitude sine wave)
        if (wake_word_counter > 0) {
            float freq = 440.0f; // A4 note
            sample += static_cast<int16_t>(8000.0f * sin(2.0f * M_PI * freq * phase));
            wake_word_counter--;
        }
        
        // Trigger wake word simulation every ~10 seconds
        if (i == 0 && (rand() % (SAMPLE_RATE * 10)) == 0) {
            wake_word_counter = SAMPLE_RATE; // 1 second of wake word
            std::cout << "[DEBUG] Simulating wake word pattern..." << std::endl;
        }
        
        buffer[i] = sample;
        phase += 1.0f / SAMPLE_RATE;
        
        // Keep phase in reasonable range
        if (phase > 1.0f) {
            phase -= 1.0f;
        }
    }
}

// Audio capture thread (simulated)
void audio_capture_thread() {
    const size_t chunk_size = 256;
    std::vector<int16_t> temp_buffer(chunk_size);
    
    std::cout << "[MIC] Audio capture thread started" << std::endl;
    
    while (g_audio_thread_running) {
        // Generate simulated audio chunk
        generate_simulated_audio(temp_buffer.data(), chunk_size);
        
        // Add to ring buffer
        {
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            
            // Prevent buffer overflow
            while (g_audio_buffer.size() > AUDIO_BUFFER_SIZE) {
                g_audio_buffer.pop();
            }
            
            // Add new samples
            for (size_t i = 0; i < chunk_size; i++) {
                g_audio_buffer.push(temp_buffer[i]);
            }
        }
        
        // Simulate real-time audio capture timing
        std::this_thread::sleep_for(
            std::chrono::microseconds((chunk_size * 1000000) / SAMPLE_RATE)
        );
    }
    
    std::cout << "[MIC] Audio capture thread stopped" << std::endl;
}

// Initialize microphone interface
bool mic_init() {
    if (g_mic_initialized) {
        std::cout << "[MIC] Already initialized" << std::endl;
        return true;
    }
    
    std::cout << "[MIC] Initializing microphone interface..." << std::endl;
    std::cout << "[MIC] Sample rate: " << SAMPLE_RATE << " Hz" << std::endl;
    std::cout << "[MIC] Channels: " << CHANNELS << std::endl;
    std::cout << "[MIC] Bits per sample: " << BITS_PER_SAMPLE << std::endl;
    
    // Clear any existing buffer data
    {
        std::lock_guard<std::mutex> lock(g_buffer_mutex);
        while (!g_audio_buffer.empty()) {
            g_audio_buffer.pop();
        }
    }
    
    // Start audio capture thread
    g_audio_thread_running = true;
    g_audio_thread = std::thread(audio_capture_thread);
    
    // Wait briefly for thread to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    g_mic_initialized = true;
    std::cout << "[MIC] Microphone initialized successfully (simulated)" << std::endl;
    
    return true;
}

// Read audio frame from microphone buffer
bool mic_read_frame(int16_t* buffer, size_t len) {
    if (!g_mic_initialized) {
        std::cerr << "[MIC] Error: Microphone not initialized" << std::endl;
        return false;
    }
    
    if (!buffer || len == 0) {
        std::cerr << "[MIC] Error: Invalid buffer or length" << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    
    // Check if we have enough samples
    if (g_audio_buffer.size() < len) {
        // Fill with zeros if not enough data (underrun)
        size_t available = g_audio_buffer.size();
        
        // Copy available samples
        for (size_t i = 0; i < available; i++) {
            buffer[i] = g_audio_buffer.front();
            g_audio_buffer.pop();
        }
        
        // Pad with silence
        for (size_t i = available; i < len; i++) {
            buffer[i] = 0;
        }
        
        return true;
    }
    
    // Copy requested samples
    for (size_t i = 0; i < len; i++) {
        buffer[i] = g_audio_buffer.front();
        g_audio_buffer.pop();
    }
    
    return true;
}

// Cleanup function (optional, called automatically on program exit)
void mic_cleanup() {
    if (!g_mic_initialized) {
        return;
    }
    
    std::cout << "[MIC] Shutting down microphone interface..." << std::endl;
    
    // Stop audio thread
    g_audio_thread_running = false;
    if (g_audio_thread.joinable()) {
        g_audio_thread.join();
    }
    
    // Clear buffer
    {
        std::lock_guard<std::mutex> lock(g_buffer_mutex);
        while (!g_audio_buffer.empty()) {
            g_audio_buffer.pop();
        }
    }
    
    g_mic_initialized = false;
    std::cout << "[MIC] Microphone interface shut down" << std::endl;
}

// Automatic cleanup on program exit
class MicCleanup {
public:
    ~MicCleanup() {
        mic_cleanup();
    }
};

static MicCleanup g_mic_cleanup;