#pragma once

#include <cstdint>
#include <array>

namespace cmx {
namespace graph {

/// @brief Graph format version for compatibility
enum class GraphFormatVersion : uint16_t {
    VERSION_1_0 = 0x0100,
    VERSION_1_1 = 0x0101,
    CURRENT = VERSION_1_1
};

/// @brief Tensor data layout formats
enum class DataLayout : uint8_t {
    UNKNOWN = 0,
    NHWC = 1,    // Batch, Height, Width, Channel
    NCHW = 2,    // Batch, Channel, Height, Width  
    HWC = 3,     // Height, Width, Channel
    CHW = 4,     // Channel, Height, Width
    NC = 5,      // Batch, Channel (fully connected)
    NHW = 6,     // Batch, Height, Width (no channel)
    SCALAR = 7   // Single value
};

/// @brief Tensor data precision types
enum class TensorPrecision : uint8_t {
    UNKNOWN = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    INT32 = 3,
    INT16 = 4,
    INT8 = 5,
    UINT8 = 6,
    BOOL = 7,
    BFLOAT16 = 8
};

/// @brief Tensor memory alignment requirements
enum class TensorAlignment : uint8_t {
    BYTE_1 = 1,
    BYTE_2 = 2,
    BYTE_4 = 4,
    BYTE_8 = 8,
    BYTE_16 = 16,
    BYTE_32 = 32
};

/// @brief Graph execution modes
enum class ExecutionMode : uint8_t {
    INFERENCE = 0,
    TRAINING = 1,
    QUANTIZATION = 2
};

/// @brief Graph optimization levels
enum class OptimizationLevel : uint8_t {
    NONE = 0,
    BASIC = 1,
    AGGRESSIVE = 2,
    MAXIMUM = 3
};

/// @brief IO tensor descriptor
struct CMXIOInfo {
    static constexpr size_t MAX_NAME_LENGTH = 64;
    static constexpr size_t MAX_DIMS = 8;
    
    char name[MAX_NAME_LENGTH];
    uint32_t tensor_id;
    DataLayout layout;
    TensorPrecision precision;
    TensorAlignment alignment;
    uint8_t rank;
    uint32_t dims[MAX_DIMS];
    uint32_t byte_size;
    uint32_t offset;  // Offset in memory buffer
    
    CMXIOInfo() : tensor_id(0), layout(DataLayout::UNKNOWN), precision(TensorPrecision::UNKNOWN),
                  alignment(TensorAlignment::BYTE_1), rank(0), byte_size(0), offset(0) {
        name[0] = '\0';
        for (size_t i = 0; i < MAX_DIMS; ++i) {
            dims[i] = 0;
        }
    }
    
    /// @brief Calculate total elements in tensor
    size_t total_elements() const {
        size_t total = 1;
        for (size_t i = 0; i < rank; ++i) {
            total *= dims[i];
        }
        return total;
    }
    
    /// @brief Get size of single element in bytes
    size_t element_size() const {
        switch (precision) {
            case TensorPrecision::FLOAT32: return 4;
            case TensorPrecision::FLOAT16: return 2;
            case TensorPrecision::BFLOAT16: return 2;
            case TensorPrecision::INT32: return 4;
            case TensorPrecision::INT16: return 2;
            case TensorPrecision::INT8: return 1;
            case TensorPrecision::UINT8: return 1;
            case TensorPrecision::BOOL: return 1;
            default: return 1;
        }
    }
    
    /// @brief Validate IO info consistency
    bool is_valid() const {
        return rank > 0 && precision != TensorPrecision::UNKNOWN && 
               layout != DataLayout::UNKNOWN && byte_size > 0;
    }
};

/// @brief Graph metadata header
struct CMXGraphHeader {
    static constexpr uint32_t MAGIC_NUMBER = 0x434D5847; // 'CMXG'
    
    uint32_t magic;
    GraphFormatVersion version;
    uint16_t flags;
    uint32_t graph_size;
    uint32_t node_count;
    uint32_t tensor_count;
    uint32_t input_count;
    uint32_t output_count;
    OptimizationLevel optimization_level;
    ExecutionMode execution_mode;
    uint16_t reserved;
    uint32_t checksum;
    
    CMXGraphHeader() : magic(MAGIC_NUMBER), version(GraphFormatVersion::CURRENT), flags(0),
                       graph_size(0), node_count(0), tensor_count(0), input_count(0), output_count(0),
                       optimization_level(OptimizationLevel::NONE), execution_mode(ExecutionMode::INFERENCE),
                       reserved(0), checksum(0) {}
    
    /// @brief Validate header magic and version
    bool is_valid() const {
        return magic == MAGIC_NUMBER && version <= GraphFormatVersion::CURRENT;
    }
};

/// @brief Graph format utilities
class CMXGraphFormat {
public:
    /// @brief Get string representation of data layout
    static const char* layout_to_string(DataLayout layout);
    
    /// @brief Get string representation of tensor precision
    static const char* precision_to_string(TensorPrecision precision);
    
    /// @brief Convert string to data layout
    static DataLayout string_to_layout(const char* str);
    
    /// @brief Convert string to tensor precision
    static TensorPrecision string_to_precision(const char* str);
    
    /// @brief Get alignment requirement for precision
    static TensorAlignment get_alignment_for_precision(TensorPrecision precision);
    
    /// @brief Calculate aligned offset
    static uint32_t align_offset(uint32_t offset, TensorAlignment alignment);
    
    /// @brief Validate layout compatibility with tensor rank
    static bool is_layout_compatible(DataLayout layout, uint8_t rank);
    
    /// @brief Get expected rank for layout
    static uint8_t get_expected_rank(DataLayout layout);
    
    /// @brief Calculate CRC32 checksum
    static uint32_t calculate_checksum(const void* data, size_t size);
    
    /// @brief Memory layout strides calculation
    static void calculate_strides(const uint32_t* dims, uint8_t rank, DataLayout layout, 
                                 uint32_t* strides);
    
    /// @brief Get format version string
    static const char* version_to_string(GraphFormatVersion version);
    
private:
    static constexpr std::array<const char*, 8> LAYOUT_NAMES = {
        "UNKNOWN", "NHWC", "NCHW", "HWC", "CHW", "NC", "NHW", "SCALAR"
    };
    
    static constexpr std::array<const char*, 9> PRECISION_NAMES = {
        "UNKNOWN", "FLOAT32", "FLOAT16", "INT32", "INT16", "INT8", "UINT8", "BOOL", "BFLOAT16"
    };
    
    static constexpr std::array<uint8_t, 8> LAYOUT_RANKS = {
        0, 4, 4, 3, 3, 2, 3, 1  // Expected ranks for each layout
    };
    
    static constexpr uint32_t CRC32_POLYNOMIAL = 0xEDB88320;
    static uint32_t crc32_table[256];
    static bool crc32_table_initialized;
    
    static void initialize_crc32_table();
};

} // namespace graph
} // namespace cmx