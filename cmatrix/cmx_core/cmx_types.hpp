#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace cmx {

// Fundamental type aliases for consistency
using cmx_i8 = int8_t;
using cmx_i16 = int16_t;
using cmx_i32 = int32_t;
using cmx_i64 = int64_t;
using cmx_u8 = uint8_t;
using cmx_u16 = uint16_t;
using cmx_u32 = uint32_t;
using cmx_u64 = uint64_t;
using cmx_f32 = float;
using cmx_f64 = double;
using cmx_size = size_t;

// Data type enumeration
enum class cmx_dtype : cmx_u8 {
    FLOAT32 = 0,
    FLOAT64 = 1,
    INT8 = 2,
    INT16 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6,
    UINT16 = 7,
    UINT32 = 8,
    UINT64 = 9,
    BOOL = 10,
    UNKNOWN = 255
};

// Memory layout enumeration
enum class cmx_layout : cmx_u8 {
    ROW_MAJOR = 0,
    COLUMN_MAJOR = 1,
    BLOCKED = 2
};

// Device type enumeration
enum class cmx_device : cmx_u8 {
    CPU = 0,
    GPU = 1,
    NPU = 2,
    FPGA = 3
};

// Tensor shape structure
struct cmx_shape {
    std::vector<cmx_size> dims;
    
    cmx_shape() = default;
    cmx_shape(std::initializer_list<cmx_size> init_dims) : dims(init_dims) {}
    explicit cmx_shape(const std::vector<cmx_size>& init_dims) : dims(init_dims) {}
    
    cmx_size rank() const { return dims.size(); }
    cmx_size total_size() const;
    bool is_scalar() const { return dims.empty(); }
    bool is_vector() const { return dims.size() == 1; }
    bool is_matrix() const { return dims.size() == 2; }
    
    cmx_size& operator[](cmx_size idx) { return dims[idx]; }
    const cmx_size& operator[](cmx_size idx) const { return dims[idx]; }
    
    bool operator==(const cmx_shape& other) const;
    bool operator!=(const cmx_shape& other) const { return !(*this == other); }
};

// Memory alignment constants
constexpr cmx_size CMX_ALIGN_BYTES = 32;  // 256-bit alignment for SIMD
constexpr cmx_size CMX_CACHE_LINE = 64;   // Typical cache line size

// Version information
constexpr cmx_u32 CMX_VERSION_MAJOR = 1;
constexpr cmx_u32 CMX_VERSION_MINOR = 0;
constexpr cmx_u32 CMX_VERSION_PATCH = 0;

// Utility functions
cmx_size cmx_dtype_size(cmx_dtype dtype);
const char* cmx_dtype_name(cmx_dtype dtype);
const char* cmx_layout_name(cmx_layout layout);
const char* cmx_device_name(cmx_device device);

// Memory alignment utilities
cmx_size cmx_align_size(cmx_size size, cmx_size alignment = CMX_ALIGN_BYTES);
bool cmx_is_aligned(const void* ptr, cmx_size alignment = CMX_ALIGN_BYTES);

} // namespace cmx