#include "cmx_cortex_m_port.hpp"
#include <cstdint>
#include <cstring>

namespace cmx {
namespace platform {
namespace cortex_m {

/**
 * @brief DMA transfer completion callback function type
 */
using DmaCallback = void(*)(void* user_data);

/**
 * @brief DMA channel configuration
 */
struct DmaConfig {
    uint8_t controller;     ///< DMA controller number (0, 1, etc.)
    uint8_t stream;         ///< DMA stream/channel number
    uint8_t priority;       ///< Transfer priority (0-3)
    bool use_interrupts;    ///< Enable transfer completion interrupts
};

/**
 * @brief DMA transfer descriptor
 */
struct DmaTransfer {
    void* src;              ///< Source address
    void* dst;              ///< Destination address
    size_t size;            ///< Transfer size in bytes
    DmaCallback callback;   ///< Completion callback (optional)
    void* user_data;        ///< User data for callback
    bool active;            ///< Transfer active flag
};

namespace {
    // DMA channel states
    constexpr size_t MAX_DMA_CHANNELS = 8;
    DmaTransfer g_dma_transfers[MAX_DMA_CHANNELS] = {};
    bool g_dma_initialized = false;
    
    // Platform-specific DMA register addresses (example for STM32F4)
    constexpr uint32_t DMA1_BASE = 0x40026000;
    constexpr uint32_t DMA2_BASE = 0x40026400;
    constexpr uint32_t DMA_STREAM_OFFSET = 0x18;
    
    // DMA register offsets
    constexpr uint32_t DMA_SxCR_OFFSET = 0x00;    // Control register
    constexpr uint32_t DMA_SxNDTR_OFFSET = 0x04;  // Number of data register
    constexpr uint32_t DMA_SxPAR_OFFSET = 0x08;   // Peripheral address register
    constexpr uint32_t DMA_SxM0AR_OFFSET = 0x0C;  // Memory 0 address register
    
    // DMA control register bits
    constexpr uint32_t DMA_SxCR_EN = (1 << 0);        // Stream enable
    constexpr uint32_t DMA_SxCR_TCIE = (1 << 4);      // Transfer complete interrupt enable
    constexpr uint32_t DMA_SxCR_DIR_M2M = (2 << 6);   // Memory-to-memory direction
    constexpr uint32_t DMA_SxCR_MINC = (1 << 10);     // Memory increment mode
    constexpr uint32_t DMA_SxCR_PINC = (1 << 9);      // Peripheral increment mode
    
    /**
     * @brief Get DMA stream base address
     */
    uint32_t get_dma_stream_base(uint8_t controller, uint8_t stream) {
        uint32_t dma_base = (controller == 0) ? DMA1_BASE : DMA2_BASE;
        return dma_base + (stream * DMA_STREAM_OFFSET) + 0x10;
    }
    
    /**
     * @brief Configure DMA stream registers
     */
    void configure_dma_stream(uint8_t controller, uint8_t stream, 
                             const DmaTransfer& transfer) {
        uint32_t stream_base = get_dma_stream_base(controller, stream);
        
        // Disable stream first
        volatile uint32_t* cr_reg = (volatile uint32_t*)(stream_base + DMA_SxCR_OFFSET);
        *cr_reg &= ~DMA_SxCR_EN;
        
        // Wait for stream to be disabled
        while (*cr_reg & DMA_SxCR_EN);
        
        // Configure control register
        uint32_t cr_value = DMA_SxCR_DIR_M2M | DMA_SxCR_MINC | DMA_SxCR_PINC;
        if (transfer.callback) {
            cr_value |= DMA_SxCR_TCIE;  // Enable transfer complete interrupt
        }
        *cr_reg = cr_value;
        
        // Set addresses and size
        *((volatile uint32_t*)(stream_base + DMA_SxPAR_OFFSET)) = (uint32_t)transfer.src;
        *((volatile uint32_t*)(stream_base + DMA_SxM0AR_OFFSET)) = (uint32_t)transfer.dst;
        *((volatile uint32_t*)(stream_base + DMA_SxNDTR_OFFSET)) = transfer.size;
    }
    
    /**
     * @brief Start DMA transfer
     */
    void start_dma_transfer(uint8_t controller, uint8_t stream) {
        uint32_t stream_base = get_dma_stream_base(controller, stream);
        volatile uint32_t* cr_reg = (volatile uint32_t*)(stream_base + DMA_SxCR_OFFSET);
        
        // Enable stream
        *cr_reg |= DMA_SxCR_EN;
    }
}

/**
 * @brief Initialize DMA subsystem
 */
bool dma_init() {
    if (g_dma_initialized) {
        return true;
    }
    
    // Enable DMA controller clocks (platform specific)
    // Example for STM32: RCC->AHB1ENR |= RCC_AHB1ENR_DMA1EN | RCC_AHB1ENR_DMA2EN;
    
    // Reset all transfer descriptors
    for (size_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        g_dma_transfers[i] = {};
    }
    
    // Configure NVIC for DMA interrupts
    // Platform-specific interrupt setup would go here
    
    g_dma_initialized = true;
    return true;
}

/**
 * @brief Deinitialize DMA subsystem
 */
void dma_deinit() {
    if (!g_dma_initialized) return;
    
    // Disable all active transfers
    for (size_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (g_dma_transfers[i].active) {
            // Stop transfer and disable stream
            uint32_t stream_base = get_dma_stream_base(i / 8, i % 8);
            volatile uint32_t* cr_reg = (volatile uint32_t*)(stream_base + DMA_SxCR_OFFSET);
            *cr_reg &= ~DMA_SxCR_EN;
        }
    }
    
    g_dma_initialized = false;
}

/**
 * @brief Synchronous DMA memory copy
 * @param dst Destination buffer
 * @param src Source buffer  
 * @param size Number of bytes to copy
 * @return True if copy completed successfully
 */
bool cmx_dma_copy_sync(void* dst, const void* src, size_t size) {
    if (!g_dma_initialized || !dst || !src || size == 0) {
        return false;
    }
    
    // Find available DMA channel
    size_t channel = MAX_DMA_CHANNELS;
    uint32_t mask = enter_critical();
    
    for (size_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (!g_dma_transfers[i].active) {
            channel = i;
            g_dma_transfers[i].active = true;
            break;
        }
    }
    
    exit_critical(mask);
    
    if (channel >= MAX_DMA_CHANNELS) {
        // No available channels - fall back to memcpy
        memcpy(dst, src, size);
        return true;
    }
    
    // Configure transfer
    DmaTransfer& transfer = g_dma_transfers[channel];
    transfer.src = const_cast<void*>(src);
    transfer.dst = dst;
    transfer.size = size;
    transfer.callback = nullptr;
    transfer.user_data = nullptr;
    
    uint8_t controller = channel / 8;
    uint8_t stream = channel % 8;
    
    // Configure and start DMA
    configure_dma_stream(controller, stream, transfer);
    start_dma_transfer(controller, stream);
    
    // Wait for completion (polling)
    uint32_t stream_base = get_dma_stream_base(controller, stream);
    volatile uint32_t* ndtr_reg = (volatile uint32_t*)(stream_base + DMA_SxNDTR_OFFSET);
    
    while (*ndtr_reg > 0) {
        // Busy wait - could yield to scheduler here in RTOS environment
        __NOP();
    }
    
    // Clear transfer descriptor
    transfer.active = false;
    
    return true;
}

/**
 * @brief Asynchronous DMA memory copy
 * @param dst Destination buffer
 * @param src Source buffer
 * @param size Number of bytes to copy
 * @param callback Completion callback function
 * @param user_data User data passed to callback
 * @return Transfer handle (channel index) or -1 if failed
 */
int cmx_dma_copy_async(void* dst, const void* src, size_t size, 
                       DmaCallback callback, void* user_data) {
    if (!g_dma_initialized || !dst || !src || size == 0) {
        return -1;
    }
    
    // Find available DMA channel
    size_t channel = MAX_DMA_CHANNELS;
    uint32_t mask = enter_critical();
    
    for (size_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (!g_dma_transfers[i].active) {
            channel = i;
            g_dma_transfers[i].active = true;
            break;
        }
    }
    
    exit_critical(mask);
    
    if (channel >= MAX_DMA_CHANNELS) {
        return -1;  // No available channels
    }
    
    // Configure transfer
    DmaTransfer& transfer = g_dma_transfers[channel];
    transfer.src = const_cast<void*>(src);
    transfer.dst = dst;
    transfer.size = size;
    transfer.callback = callback;
    transfer.user_data = user_data;
    
    uint8_t controller = channel / 8;
    uint8_t stream = channel % 8;
    
    // Configure and start DMA
    configure_dma_stream(controller, stream, transfer);
    start_dma_transfer(controller, stream);
    
    return static_cast<int>(channel);
}

/**
 * @brief Check if DMA transfer is complete
 * @param handle Transfer handle returned by cmx_dma_copy_async
 * @return True if transfer is complete
 */
bool cmx_dma_is_complete(int handle) {
    if (handle < 0 || handle >= static_cast<int>(MAX_DMA_CHANNELS)) {
        return true;  // Invalid handle
    }
    
    return !g_dma_transfers[handle].active;
}

/**
 * @brief Cancel ongoing DMA transfer
 * @param handle Transfer handle to cancel
 * @return True if transfer was cancelled successfully
 */
bool cmx_dma_cancel(int handle) {
    if (handle < 0 || handle >= static_cast<int>(MAX_DMA_CHANNELS)) {
        return false;
    }
    
    DmaTransfer& transfer = g_dma_transfers[handle];
    if (!transfer.active) {
        return true;  // Already complete
    }
    
    uint8_t controller = handle / 8;
    uint8_t stream = handle % 8;
    
    // Disable DMA stream
    uint32_t stream_base = get_dma_stream_base(controller, stream);
    volatile uint32_t* cr_reg = (volatile uint32_t*)(stream_base + DMA_SxCR_OFFSET);
    *cr_reg &= ~DMA_SxCR_EN;
    
    // Wait for stream to be disabled
    while (*cr_reg & DMA_SxCR_EN);
    
    // Clear transfer descriptor
    transfer.active = false;
    
    return true;
}

/**
 * @brief DMA interrupt handler for transfer completion
 * This should be called from the platform-specific DMA interrupt handlers
 */
extern "C" void dma_transfer_complete_handler(uint8_t controller, uint8_t stream) {
    size_t channel = controller * 8 + stream;
    
    if (channel >= MAX_DMA_CHANNELS) {
        return;
    }
    
    DmaTransfer& transfer = g_dma_transfers[channel];
    if (!transfer.active) {
        return;
    }
    
    // Clear interrupt flags (platform specific)
    // Example: Clear transfer complete interrupt flag
    
    // Call user callback if provided
    if (transfer.callback) {
        transfer.callback(transfer.user_data);
    }
    
    // Mark transfer as complete
    transfer.active = false;
}

/**
 * @brief Get DMA transfer statistics
 * @param handle Transfer handle
 * @param bytes_remaining Output: bytes remaining to transfer
 * @return True if handle is valid and transfer is active
 */
bool cmx_dma_get_status(int handle, size_t* bytes_remaining) {
    if (handle < 0 || handle >= static_cast<int>(MAX_DMA_CHANNELS) || !bytes_remaining) {
        return false;
    }
    
    const DmaTransfer& transfer = g_dma_transfers[handle];
    if (!transfer.active) {
        *bytes_remaining = 0;
        return false;
    }
    
    uint8_t controller = handle / 8;
    uint8_t stream = handle % 8;
    
    // Read remaining count from DMA register
    uint32_t stream_base = get_dma_stream_base(controller, stream);
    volatile uint32_t* ndtr_reg = (volatile uint32_t*)(stream_base + DMA_SxNDTR_OFFSET);
    
    *bytes_remaining = *ndtr_reg;
    return true;
}

/**
 * @brief Get number of available DMA channels
 * @return Number of free DMA channels
 */
size_t cmx_dma_get_free_channels() {
    size_t free_count = 0;
    
    uint32_t mask = enter_critical();
    for (size_t i = 0; i < MAX_DMA_CHANNELS; i++) {
        if (!g_dma_transfers[i].active) {
            free_count++;
        }
    }
    exit_critical(mask);
    
    return free_count;
}

// Platform-specific DMA interrupt handlers
// These should be implemented for your specific MCU

extern "C" __attribute__((weak)) void DMA1_Stream0_IRQHandler() {
    dma_transfer_complete_handler(0, 0);
}

extern "C" __attribute__((weak)) void DMA1_Stream1_IRQHandler() {
    dma_transfer_complete_handler(0, 1);
}

extern "C" __attribute__((weak)) void DMA1_Stream2_IRQHandler() {
    dma_transfer_complete_handler(0, 2);
}

extern "C" __attribute__((weak)) void DMA1_Stream3_IRQHandler() {
    dma_transfer_complete_handler(0, 3);
}

extern "C" __attribute__((weak)) void DMA1_Stream4_IRQHandler() {
    dma_transfer_complete_handler(0, 4);
}

extern "C" __attribute__((weak)) void DMA1_Stream5_IRQHandler() {
    dma_transfer_complete_handler(0, 5);
}

extern "C" __attribute__((weak)) void DMA1_Stream6_IRQHandler() {
    dma_transfer_complete_handler(0, 6);
}

extern "C" __attribute__((weak)) void DMA1_Stream7_IRQHandler() {
    dma_transfer_complete_handler(0, 7);
}

extern "C" __attribute__((weak)) void DMA2_Stream0_IRQHandler() {
    dma_transfer_complete_handler(1, 0);
}

extern "C" __attribute__((weak)) void DMA2_Stream1_IRQHandler() {
    dma_transfer_complete_handler(1, 1);
}

extern "C" __attribute__((weak)) void DMA2_Stream2_IRQHandler() {
    dma_transfer_complete_handler(1, 2);
}

extern "C" __attribute__((weak)) void DMA2_Stream3_IRQHandler() {
    dma_transfer_complete_handler(1, 3);
}

extern "C" __attribute__((weak)) void DMA2_Stream4_IRQHandler() {
    dma_transfer_complete_handler(1, 4);
}

extern "C" __attribute__((weak)) void DMA2_Stream5_IRQHandler() {
    dma_transfer_complete_handler(1, 5);
}

extern "C" __attribute__((weak)) void DMA2_Stream6_IRQHandler() {
    dma_transfer_complete_handler(1, 6);
}

extern "C" __attribute__((weak)) void DMA2_Stream7_IRQHandler() {
    dma_transfer_complete_handler(1, 7);
}

} // namespace cortex_m
} // namespace platform
} // namespace cmx