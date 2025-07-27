# Gesture Recognition Demo

A real-time gesture recognition system for embedded microcontrollers using the `cmatrix` ML runtime. This demo processes IMU sensor data to classify hand gestures and triggers corresponding output actions.

## Overview

This project demonstrates on-device machine learning for gesture recognition using:
- **Runtime**: CMatrix ML inference engine
- **Hardware**: Low-power MCU (STM32/ESP32) + IMU sensor
- **Model**: Pre-trained gesture classification CNN/RNN
- **Languages**: Modern embedded C++ (C++17)

## Use Case

The system continuously monitors motion data from an IMU sensor, processes it through a neural network model, and responds to detected gestures with visual/audio feedback. Perfect for:
- IoT device control via hand gestures
- Wearable fitness tracking
- Smart home automation
- Interactive embedded displays
- Accessibility interfaces

## Required Hardware

### Core Components
- **Microcontroller**: STM32F4xx series or ESP32 DevKit
- **IMU Sensor**: MPU6050 or ADXL345 (I2C interface)
- **Power Supply**: 3.3V regulated supply

### Optional Output Modules
- **LEDs**: For visual gesture feedback
- **Buzzer/Piezo**: For audio confirmation
- **Display**: OLED/LCD for gesture visualization
- **UART/USB**: For serial debugging output

### Wiring Example (STM32 + MPU6050)
```
STM32F4 Pin  | MPU6050 Pin | Function
-------------|-------------|----------
PB6          | SCL         | I2C Clock
PB7          | SDA         | I2C Data
3V3          | VCC         | Power
GND          | GND         | Ground
PB0          | -           | Status LED
PC13         | -           | Buzzer
```

## Model Information

### Gesture Classes
The trained model recognizes 5 distinct gestures:
1. **LEFT** - Left swipe motion
2. **RIGHT** - Right swipe motion  
3. **UP** - Upward motion
4. **DOWN** - Downward motion
5. **CIRCLE** - Circular hand movement

### Model Architecture
- **Type**: Convolutional Neural Network (CNN) or Recurrent Neural Network (RNN)
- **Input**: 6-axis IMU data (accelerometer + gyroscope)
- **Window Size**: 64 time steps (~1.3 seconds at 50Hz)
- **Features**: 6 channels (ax, ay, az, gx, gy, gz)
- **Output**: 5 gesture classes + confidence scores

### Training Data
- Sampling rate: 50Hz
- Window overlap: 50%
- Data augmentation: Rotation, noise, scaling
- Validation accuracy: >92%

## Build Instructions

### Prerequisites
- **Toolchain**: ARM GCC or ESP-IDF
- **IDE**: STM32CubeIDE, PlatformIO, or Arduino IDE
- **Libraries**: HAL drivers, CMatrix runtime

### STM32 Build
```bash
# Clone repository
git clone <repository-url>
cd gesture_demo

# Configure for STM32
mkdir build && cd build
cmake -DTARGET_PLATFORM=STM32 ..

# Build firmware
make -j4

# Flash to device
st-flash write gesture_demo.bin 0x8000000
```

### ESP32 Build
```bash
# Using PlatformIO
platformio init --board esp32dev
platformio run --target upload

# Or using ESP-IDF
idf.py build
idf.py flash monitor
```

## Usage Instructions

### Initial Setup
1. **Hardware Assembly**: Connect IMU sensor to MCU via I2C
2. **Power On**: Connect power supply and verify LED status
3. **Calibration**: Keep device stationary for 3 seconds after boot
4. **Serial Monitor**: Connect UART at 115200 baud for debug output

### Performing Gestures
- **Position**: Hold device naturally in hand
- **Timing**: Perform gestures within 1-2 seconds
- **Range**: Move hand in 20-30cm range for best detection
- **Pause**: Wait 0.5s between consecutive gestures

### Expected Responses

| Gesture | Visual Response | Audio Response | Serial Output |
|---------|-----------------|----------------|---------------|
| LEFT    | Single LED blink| None          | "Gesture: Left, Confidence: 0.XX" |
| RIGHT   | Double LED blink| None          | "Gesture: Right, Confidence: 0.XX" |
| UP      | LED solid (2s)  | None          | "Gesture: Up, Confidence: 0.XX" |
| DOWN    | None            | Buzzer beep   | "Gesture: Down, Confidence: 0.XX" |
| CIRCLE  | LED blink       | Short beep    | "Gesture: Circle, Confidence: 0.XX" |

## Configuration

### Sensor Settings
```cpp
// In cmx_config.hpp
#define IMU_SAMPLE_RATE_HZ      50
#define GESTURE_WINDOW_SIZE     64
#define CONFIDENCE_THRESHOLD    0.7f
```

### Pin Assignments
```cpp
// Default pin mapping
#define STATUS_LED_PIN    GPIO_PIN_0    // PB0
#define BUZZER_PIN        GPIO_PIN_13   // PC13
#define I2C_SCL_PIN       GPIO_PIN_6    // PB6
#define I2C_SDA_PIN       GPIO_PIN_7    // PB7
```

## Troubleshooting

### Common Issues
- **No Response**: Check I2C connections and power supply
- **False Positives**: Adjust confidence threshold or recalibrate
- **Poor Accuracy**: Ensure consistent gesture timing and amplitude
- **Serial Errors**: Verify baud rate and UART pin configuration

### Debug Mode
Enable verbose logging by defining `DEBUG_GESTURES` in build configuration:
```cpp
#define DEBUG_GESTURES 1  // Enable detailed gesture logging
```

## Performance Metrics

- **Inference Time**: ~15ms per classification
- **Memory Usage**: 32KB RAM, 128KB Flash
- **Power Consumption**: 45mA active, 12μA sleep
- **Gesture Latency**: <200ms end-to-end
- **Classification Accuracy**: 89% on test dataset

## Customization

### Adding New Gestures
1. Collect training data for new gesture patterns
2. Retrain model with expanded dataset
3. Update `GestureClass` enum in `output_handler.hpp`
4. Add corresponding response in `handle_gesture()`

### Changing Output Actions
Modify the `handle_gesture()` function in `output_handler.cpp` to customize responses:
```cpp
case GestureClass::LEFT:
    // Custom action: Send WiFi command
    send_wifi_command("DEVICE_LEFT");
    break;
```

## License

This project is provided under the MIT License. See LICENSE file for details.

## Support

For technical support or questions:
- Check hardware connections and power supply
- Review serial output for error messages  
- Ensure IMU sensor is properly calibrated
- Verify model file is correctly loaded

## Version History

- **v1.0.0**: Initial release with 5-gesture classification
- **v1.1.0**: Added ESP32 support and improved accuracy
- **v1.2.0**: Enhanced power management and sleep modes
