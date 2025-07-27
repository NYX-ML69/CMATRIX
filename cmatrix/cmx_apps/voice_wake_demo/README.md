# Voice Wake Word Demo

A real-time voice wake-word detection application built using the CMatrix runtime framework. This demo integrates microphone input, machine learning inference, and response handling to detect wake words and trigger appropriate actions.

## Overview

The application continuously monitors audio input from a microphone, processes the audio through a CMatrix-deployed neural network model, and responds to wake word detections with configurable actions including visual feedback, audio alerts, and external device communication.

## Features

- **Real-time Audio Processing**: Captures and processes audio at 16kHz sample rate
- **ML-based Wake Word Detection**: Uses CMatrix runtime for efficient neural network inference
- **Configurable Response Actions**: Visual, audio, and UART/serial output responses
- **Debouncing**: Prevents rapid repeated triggers
- **Modular Architecture**: Clean separation of audio input, inference, and response handling

## Hardware Requirements

### Minimum Requirements
- **Microcontroller/SoC**: ARM Cortex-M4 or equivalent with CMatrix support
- **Memory**: 256KB RAM minimum (512KB recommended)
- **Storage**: 2MB flash memory for application and model
- **Audio Input**: I2S microphone or analog microphone with ADC

### Recommended Hardware
- **Development Board**: STM32F4 Discovery, ESP32-S3, or similar
- **Microphone**: MEMS I2S microphone (e.g., INMP441)
- **Indicators**: LED for visual feedback
- **Communication**: UART for external device integration
- **Power**: 3.3V/5V power supply

## Software Dependencies

- **CMatrix Runtime**: v2.0 or higher
- **CMake**: v3.12 or higher
- **GCC ARM Toolchain**: v9.0 or higher
- **Model File**: `wake_model.cmx` (trained wake word detection model)

## Directory Structure

```
voice_wake_demo/
├── cmx_config.hpp      # Configuration constants and parameters
├── main.cpp            # Main application entry point
├── mic_input.cpp       # Microphone input handling
├── wake_handler.cpp    # Wake word response actions
├── README.md           # This documentation
└── CMakeLists.txt      # Build configuration (not provided)
```

## Build Instructions

### 1. Prerequisites
Ensure you have the CMatrix SDK and ARM toolchain installed:

```bash
# Install CMatrix SDK (example path)
export CMATRIX_SDK_PATH=/opt/cmatrix-sdk

# Verify toolchain
arm-none-eabi-gcc --version
```

### 2. Build Process

```bash
# Clone or extract the voice_wake_demo directory
cd voice_wake_demo

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=arm-none-eabi.cmake

# Build the application
make -j4

# Output binary: voice_wake_demo.elf
```

### 3. Model Preparation
Place your trained wake word model in the build directory:

```bash
# Copy your trained model
cp /path/to/your/wake_model.cmx ./wake_model.cmx
```

## Running the Application

### 1. Flash to Target Hardware

```bash
# Using OpenOCD (example for STM32)
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
        -c "program voice_wake_demo.elf verify reset exit"

# Using ST-Link utility
st-flash write voice_wake_demo.bin 0x8000000
```

### 2. Monitor Output

Connect to UART console to see application output:

```bash
# Using screen (Linux/macOS)
screen /dev/ttyUSB0 115200

# Using PuTTY (Windows)
# Configure: Serial, COM port, 115200 baud
```

### 3. Expected Behavior

When running successfully, you should see:

```
Voice Wake Demo v1.0.0
Initializing voice wake word detection...
Microphone initialized
CMatrix runtime initialized successfully
Loaded wake word model: wake_model.cmx
Starting wake word detection...
Listening for wake words (Ctrl+C to exit)...

[When wake word is detected:]
🎯 ===== WAKE WORD DETECTED! ===== 🎯
⏰ Time: 2025-07-25 14:30:45.123
🔢 Detection count: 1

💡 [VISUAL] Wake word LED ON
🔊 [AUDIO] Wake word beep: BEEP!
📡 [UART] Sending wake signal to external device...
✅ [HANDLER] All wake word responses completed
🎯 ===== WAKE HANDLING COMPLETE ===== 🎯
```

## Configuration

### Audio Settings
Modify `cmx_config.hpp` to adjust audio parameters:

```cpp
#define SAMPLE_RATE 16000    // Audio sample rate (Hz)
#define FRAME_SIZE 1024      // Audio frame size (samples)
#define CHANNELS 1           # Number of audio channels
```

### Detection Sensitivity
Adjust wake word detection sensitivity:

```cpp
#define CONFIDENCE_THRESHOLD 0.8f    // Detection confidence (0.0-1.0)
#define DEBOUNCE_TIME_MS 2000        // Time between detections (ms)
```

### Response Actions
Customize response behavior in `wake_handler.cpp`:
- Enable/disable LED feedback
- Configure UART communication
- Add custom response actions

## Troubleshooting

### Common Issues

**1. Model Loading Failed**
```
Failed to load model from wake_model.cmx: CMX_ERROR_FILE_NOT_FOUND
```
- Ensure `wake_model.cmx` is in the correct directory
- Verify model file format compatibility with CMatrix runtime

**2. Microphone Initialization Failed**
```
Failed to initialize microphone
```
- Check hardware connections
- Verify I2S/ADC configuration
- Ensure sufficient power supply

**3. Low Detection Accuracy**
```
Wake word not detected or false positives
```
- Adjust `CONFIDENCE_THRESHOLD` in configuration
- Verify microphone positioning and audio quality
- Retrain model with better dataset if needed

**4. Memory Issues**
```
Stack overflow or allocation failures
```
- Reduce `AUDIO_BUFFER_SIZE` and `FRAME_SIZE`
- Optimize model size
- Check available RAM

### Debug Mode

Enable debug output by adding to `cmx_config.hpp`:

```cpp
#define DEBUG_MODE 1
#define VERBOSE_AUDIO 1
```

## Performance Optimization

- **Model Optimization**: Use quantized models (INT8) for better performance
- **Buffer Tuning**: Adjust audio buffer sizes for memory/latency trade-off
- **Inference Frequency**: Process every Nth frame to reduce CPU usage
- **Power Management**: Implement sleep modes between detections

## Integration Examples

### Home Automation
```cpp
void handle_wake_detected() {
    // Turn on smart lights
    send_command("lights_on");
    
    // Start voice assistant
    activate_voice_assistant();
}
```

### Security System
```cpp
void handle_wake_detected() {
    // Log security event
    log_security_event("WAKE_DETECTED");
    
    // Send alert to monitoring system
    send_security_alert();
}
```

## License

This demo application is provided as-is for educational and development purposes. Please refer to the CMatrix SDK license for runtime usage terms.

## Support

For technical support and questions:
- CMatrix SDK Documentation
- Hardware vendor support channels
- Community forums and repositories