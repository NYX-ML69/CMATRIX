#ifndef INPUT_INTERFACE_HPP
#define INPUT_INTERFACE_HPP

#include <cstdint>

// =============================================================================
// Public Interface for IMU Data Collection
// =============================================================================

/**
 * @brief Initialize the IMU sensor and data collection system
 * @return true if initialization successful, false otherwise
 */
bool initialize_imu();

/**
 * @brief Update IMU data collection (call from high-frequency task)
 * This function should be called at the configured sample rate
 */
void update_imu_data();

/**
 * @brief Check if a complete IMU window is ready for inference
 * @return true if data is ready, false otherwise
 */
bool is_imu_data_ready();

/**
 * @brief Read a complete IMU data window for inference
 * @param out_buffer Output buffer to store the windowed data
 *                   Must be allocated with size INPUT_SIZE * sizeof(float)
 * @return true if data was successfully copied, false otherwise
 */
bool read_imu_window(float* out_buffer);

/**
 * @brief Get current instantaneous IMU readings
 * @param accel_out Buffer for accelerometer data [3 floats: X,Y,Z in g]
 * @param gyro_out Buffer for gyroscope data [3 floats: X,Y,Z in deg/s]
 */
void get_current_imu_data(float* accel_out, float* gyro_out);

/**
 * @brief Get current IMU temperature reading
 * @return Temperature in degrees Celsius
 */
float get_imu_temperature();

/**
 * @brief Reset the sliding window buffer (useful for recalibration)
 */
void reset_imu_window();

/**
 * @brief Normalize input data for neural network inference
 * @param data Input data array to normalize in-place
 */
void normalize_input_data(float* data);

#endif // INPUT_INTERFACE_HPP