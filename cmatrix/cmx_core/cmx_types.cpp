// cmx_types.cpp
// CMatrix Framework Implementation
#include "cmx_types.hpp"
#include <numeric>

namespace cmx {

cmx_size cmx_shape::total_size() const {
    if (dims.empty()) return 1;  // scalar
    return std::accumulate(dims.begin(), dims.end(), cmx_size(1), std::multiplies<cmx_size>());
}

bool cmx_shape::operator==(const cmx_shape& other) const {
    return dims == other.dims;
}

cmx_size cmx_dtype_size(cmx_dtype dtype) {
    switch (dtype) {
        case cmx_dtype::FLOAT32: return sizeof(cmx_f32);
        case cmx_dtype::FLOAT64: return sizeof(cmx_f64);
        case cmx_dtype::INT8: return sizeof(cmx_i8);
        case cmx_dtype::INT16: return sizeof(cmx_i16);
        case cmx_dtype::INT32: return sizeof(cmx_i32);
        case cmx_dtype::INT64: return sizeof(cmx_i64);
        case cmx_dtype::UINT8: return sizeof(cmx_u8);
        case cmx_dtype::UINT16: return sizeof(cmx_u16);
        case cmx_dtype::UINT32: return sizeof(cmx_u32);
        case cmx_dtype::UINT64: return sizeof(cmx_u64);
        case cmx_dtype::BOOL: return sizeof(bool);
        case cmx_dtype::UNKNOWN:
        default: return 0;
    }
}

const char* cmx_dtype_name(cmx_dtype dtype) {
    switch (dtype) {
        case cmx_dtype::FLOAT32: return "float32";
        case cmx_dtype::FLOAT64: return "float64";
        case cmx_dtype::INT8: return "int8";
        case cmx_dtype::INT16: return "int16";
        case cmx_dtype::INT32: return "int32";
        case cmx_dtype::INT64: return "int64";
        case cmx_dtype::UINT8: return "uint8";
        case cmx_dtype::UINT16: return "uint16";
        case cmx_dtype::UINT32: return "uint32";
        case cmx_dtype::UINT64: return "uint64";
        case cmx_dtype::BOOL: return "bool";
        case cmx_dtype::UNKNOWN:
        default: return "unknown";
    }
}

const char* cmx_layout_name(cmx_layout layout) {
    switch (layout) {
        case cmx_layout::ROW_MAJOR: return "row_major";
        case cmx_layout::COLUMN_MAJOR: return "column_major";
        case cmx_layout::BLOCKED: return "blocked";
        default: return "unknown";
    }
}

const char* cmx_device_name(cmx_device device) {
    switch (device) {
        case cmx_device::CPU: return "cpu";
        case cmx_device::GPU: return "gpu";
        case cmx_device::NPU: return "npu";
        case cmx_device::FPGA: return "fpga";
        default: return "unknown";
    }
}

cmx_size cmx_align_size(cmx_size size, cmx_size alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

bool cmx_is_aligned(const void* ptr, cmx_size alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

} // namespace cmx
