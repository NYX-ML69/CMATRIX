#include "cmx_riscv_dma.hpp"
#include <cstring>

namespace cmx::platform::riscv {

// Static member variables
static DMAConfig channel_configs[RISCV_DMA::get_channel_count()];
static DMACallback channel_callbacks[RISCV_DMA::get_channel_count()];
static void* channel_user_data[RISCV_DMA::get_channel_count()];
static volatile DMAStatus channel_status[RISCV_DMA::get_channel_count()];

// Hardware register definitions (platform-specific)
struct DMARegisters {
    volatile uint32_t CTRL;         // Control register
    volatile uint32_t STATUS;       // Status register
    volatile uint32_t SRC_ADDR;     // Source address
    volatile uint32_t DST_ADDR;     // Destination address
    volatile uint32_t TRANSFER_SIZE; // Transfer size
    volatile uint32_t CONFIG;       // Configuration register
    volatile uint32_t RESERVED[2];   // Reserved
};

static volatile DMARegisters* get_dma_registers(uint32_t channel_id) {
    return reinterpret_cast<volatile DMARegisters*>(
        RISCV_DMA::DMA_BASE_ADDR + (channel_id * sizeof(DMARegisters))
    );
}

// Register bit definitions
static constexpr uint32_t DMA_CTRL_ENABLE     = (1 << 0);
static constexpr uint32_t DMA_CTRL_START      = (1 << 1);
static constexpr uint32_t DMA_CTRL_STOP       = (1 << 2);
static constexpr uint32_t DMA_CTRL_INT_EN     = (1 << 3);
static constexpr uint32_t DMA_CTRL_RESET      = (1 << 4);

static constexpr uint32_t DMA_STATUS_BUSY     = (1 << 0);
static constexpr uint32_t DMA_STATUS_COMPLETE = (1 << 1);
static constexpr uint32_t DMA_STATUS_ERROR    = (1 << 2);

static constexpr uint32_t DMA_CFG_SRC_INC     = (1 << 0);
static constexpr uint32_t DMA_CFG_DST_INC     = (1 << 1);
static constexpr uint32_t DMA_CFG_WIDTH_MASK  = (0x3 << 2);
static constexpr uint32_t DMA_CFG_BURST_MASK  = (0xF << 4);

bool RISCV_DMA::initialize() {
    // Initialize all channels to idle state
    for (uint32_t i = 0; i < MAX_DMA_CHANNELS; ++i) {
        channel_status[i] = DMAStatus::IDLE;
        channel_callbacks[i] = nullptr;
        channel_user_data[i] = nullptr;
        
        // Reset hardware channel
        reset_channel(i);
    }
    
    // Enable DMA controller clock (platform-specific)
    // This would typically involve writing to a clock control register
    
    return true;
}

void RISCV_DMA::deinitialize() {
    // Stop all active transfers
    for (uint32_t i = 0; i < MAX_DMA_CHANNELS; ++i) {
        if (is_busy(i)) {
            stop_transfer(i);
        }
        reset_channel(i);
    }
    
    // Disable DMA controller clock (platform-specific)
}

bool RISCV_DMA::configure_channel(const DMAConfig& config) {
    if (!validate_channel(config.channel_id) || !validate_transfer_params(config)) {
        return false;
    }
    
    uint32_t channel_id = config.channel_id;
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    
    // Stop channel if running
    if (is_busy(channel_id)) {
        stop_transfer(channel_id);
    }
    
    // Store configuration
    channel_configs[channel_id] = config;
    
    // Configure hardware registers
    regs->SRC_ADDR = config.src_addr;
    regs->DST_ADDR = config.dst_addr;
    regs->TRANSFER_SIZE = config.transfer_size;
    
    // Build configuration register
    uint32_t cfg_reg = 0;
    if (config.src_increment) cfg_reg |= DMA_CFG_SRC_INC;
    if (config.dst_increment) cfg_reg |= DMA_CFG_DST_INC;
    
    // Set data width
    cfg_reg |= ((config.data_width - 1) << 2) & DMA_CFG_WIDTH_MASK;
    
    // Set burst size
    cfg_reg |= calculate_burst_config(config.burst_size) & DMA_CFG_BURST_MASK;
    
    regs->CONFIG = cfg_reg;
    
    // Configure control register
    uint32_t ctrl_reg = DMA_CTRL_ENABLE;
    if (config.enable_interrupt) {
        ctrl_reg |= DMA_CTRL_INT_EN;
    }
    regs->CTRL = ctrl_reg;
    
    channel_status[channel_id] = DMAStatus::IDLE;
    return true;
}

bool RISCV_DMA::start_transfer(uint32_t channel_id, DMADirection direction,
                               DMACallback callback, void* user_data) {
    if (!validate_channel(channel_id) || is_busy(channel_id)) {
        return false;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    
    // Store callback and user data
    channel_callbacks[channel_id] = callback;
    channel_user_data[channel_id] = user_data;
    
    // Update status
    channel_status[channel_id] = DMAStatus::BUSY;
    
    // Start transfer
    uint32_t ctrl_reg = regs->CTRL;
    ctrl_reg |= DMA_CTRL_START;
    regs->CTRL = ctrl_reg;
    
    return true;
}

bool RISCV_DMA::stop_transfer(uint32_t channel_id) {
    if (!validate_channel(channel_id)) {
        return false;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    
    // Send stop command
    uint32_t ctrl_reg = regs->CTRL;
    ctrl_reg |= DMA_CTRL_STOP;
    regs->CTRL = ctrl_reg;
    
    // Wait for stop to take effect
    uint32_t timeout = 1000;
    while ((regs->STATUS & DMA_STATUS_BUSY) && --timeout) {
        // Small delay
        for (volatile int i = 0; i < 100; ++i);
    }
    
    channel_status[channel_id] = DMAStatus::IDLE;
    return timeout > 0;
}

bool RISCV_DMA::is_busy(uint32_t channel_id) {
    if (!validate_channel(channel_id)) {
        return false;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    return (regs->STATUS & DMA_STATUS_BUSY) != 0;
}

DMAStatus RISCV_DMA::get_status(uint32_t channel_id) {
    if (!validate_channel(channel_id)) {
        return DMAStatus::ERROR;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    uint32_t hw_status = regs->STATUS;
    
    if (hw_status & DMA_STATUS_ERROR) {
        channel_status[channel_id] = DMAStatus::ERROR;
    } else if (hw_status & DMA_STATUS_COMPLETE) {
        channel_status[channel_id] = DMAStatus::COMPLETE;
    } else if (hw_status & DMA_STATUS_BUSY) {
        channel_status[channel_id] = DMAStatus::BUSY;
    } else {
        channel_status[channel_id] = DMAStatus::IDLE;
    }
    
    return channel_status[channel_id];
}

bool RISCV_DMA::wait_for_completion(uint32_t channel_id, uint32_t timeout_ms) {
    if (!validate_channel(channel_id)) {
        return false;
    }
    
    if (timeout_ms == 0) {
        timeout_ms = DMA_TIMEOUT_MS;
    }
    
    uint32_t start_time = 0; // Would get from timer in real implementation
    
    while (is_busy(channel_id)) {
        // Check timeout (simplified - would use real timer)
        if (timeout_ms > 0) {
            uint32_t current_time = 0; // Would get from timer
            if ((current_time - start_time) > timeout_ms) {
                return false;
            }
        }
        
        // Small delay to prevent busy waiting
        for (volatile int i = 0; i < 1000; ++i);
    }
    
    return get_status(channel_id) == DMAStatus::COMPLETE;
}

bool RISCV_DMA::memcpy_dma(void* dst, const void* src, size_t size, uint32_t channel_id) {
    if (!dst || !src || size == 0 || !validate_channel(channel_id)) {
        return false;
    }
    
    // Ensure addresses are properly aligned
    if (!is_dma_aligned(src) || !is_dma_aligned(dst)) {
        // Fallback to regular memcpy for unaligned transfers
        std::memcpy(dst, src, size);
        return true;
    }
    
    // Configure DMA for memory-to-memory transfer
    DMAConfig config = {};
    config.channel_id = channel_id;
    config.src_addr = reinterpret_cast<uint32_t>(src);
    config.dst_addr = reinterpret_cast<uint32_t>(dst);
    config.transfer_size = size;
    config.src_increment = 1;
    config.dst_increment = 1;
    config.data_width = 4; // 32-bit transfers for better performance
    config.burst_size = 8;
    config.enable_interrupt = false;
    
    if (!configure_channel(config)) {
        return false;
    }
    
    if (!start_transfer(channel_id, DMADirection::MEM_TO_MEM)) {
        return false;
    }
    
    return wait_for_completion(channel_id);
}

bool RISCV_DMA::memset_dma(void* dst, uint8_t value, size_t size, uint32_t channel_id) {
    if (!dst || size == 0 || !validate_channel(channel_id)) {
        return false;
    }
    
    // For memset, we need a source buffer with the pattern
    // In a real implementation, this might use a dedicated hardware feature
    // or a small pattern buffer that gets repeated
    
    // Fallback to regular memset for now
    std::memset(dst, value, size);
    return true;
}

void RISCV_DMA::reset_channel(uint32_t channel_id) {
    if (!validate_channel(channel_id)) {
        return;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    
    // Reset the channel
    regs->CTRL = DMA_CTRL_RESET;
    
    // Wait for reset to complete
    for (volatile int i = 0; i < 1000; ++i);
    
    // Clear all registers
    regs->CTRL = 0;
    regs->SRC_ADDR = 0;
    regs->DST_ADDR = 0;
    regs->TRANSFER_SIZE = 0;
    regs->CONFIG = 0;
    
    channel_status[channel_id] = DMAStatus::IDLE;
    channel_callbacks[channel_id] = nullptr;
    channel_user_data[channel_id] = nullptr;
}

void RISCV_DMA::set_interrupt_enable(uint32_t channel_id, bool enable) {
    if (!validate_channel(channel_id)) {
        return;
    }
    
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    
    uint32_t ctrl_reg = regs->CTRL;
    if (enable) {
        ctrl_reg |= DMA_CTRL_INT_EN;
    } else {
        ctrl_reg &= ~DMA_CTRL_INT_EN;
    }
    regs->CTRL = ctrl_reg;
}

// Private helper functions
bool RISCV_DMA::validate_channel(uint32_t channel_id) {
    return channel_id < MAX_DMA_CHANNELS;
}

bool RISCV_DMA::validate_transfer_params(const DMAConfig& config) {
    // Check alignment requirements
    if (config.data_width > 1) {
        uint32_t alignment = config.data_width;
        if ((config.src_addr & (alignment - 1)) != 0 ||
            (config.dst_addr & (alignment - 1)) != 0) {
            return false;
        }
    }
    
    // Check transfer size
    if (config.transfer_size == 0 || config.transfer_size > 0xFFFF) {
        return false;
    }
    
    // Check data width
    if (config.data_width != 1 && config.data_width != 2 && config.data_width != 4) {
        return false;
    }
    
    // Check burst size
    if (config.burst_size > 16) {
        return false;
    }
    
    return true;
}

void RISCV_DMA::handle_dma_interrupt(uint32_t channel_id) {
    if (!validate_channel(channel_id)) {
        return;
    }
    
    DMAStatus status = get_status(channel_id);
    
    // Call user callback if registered
    if (channel_callbacks[channel_id]) {
        channel_callbacks[channel_id](channel_id, status, channel_user_data[channel_id]);
    }
    
    // Clear interrupt flag (platform-specific)
    volatile DMARegisters* regs = get_dma_registers(channel_id);
    regs->STATUS = DMA_STATUS_COMPLETE | DMA_STATUS_ERROR;
}

uint32_t RISCV_DMA::calculate_burst_config(uint32_t burst_size) {
    // Convert burst size to hardware configuration
    if (burst_size <= 1) return 0;
    if (burst_size <= 2) return 1;
    if (burst_size <= 4) return 2;
    if (burst_size <= 8) return 3;
    return 4; // 16-burst
}

} // namespace cmx::platform::riscv