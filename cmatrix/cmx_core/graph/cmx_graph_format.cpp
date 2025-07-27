#include "cmx_graph_format.hpp"
#include <cstring>
#include <algorithm>

namespace cmx {
namespace graph {

uint32_t CMXGraphFormat::crc32_table[256];
bool CMXGraphFormat::crc32_table_initialized = false;

const char* CMXGraphFormat::layout_to_string(DataLayout layout) {
    size_t index = static_cast<size_t>(layout);
    if (index < LAYOUT_NAMES.size()) {
        return LAYOUT_NAMES[index];
    }
    return "UNKNOWN";
}

const char* CMXGraphFormat::precision_to_string(TensorPrecision precision) {
    size_t index = static_cast<size_t>(precision);
    if (index < PRECISION_NAMES.size()) {
        return PRECISION_NAMES[index];
    }
    return "UNKNOWN";
}

DataLayout CMXGraphFormat::string_to_layout(const char* str) {
    if (!str) return DataLayout::UNKNOWN;
    
    for (size_t i = 0; i < LAYOUT_NAMES.size(); ++i) {
        if (strcmp(str, LAYOUT_NAMES[i]) == 0) {
            return static_cast<DataLayout>(i);
        }
    }
    return DataLayout::UNKNOWN;
}

TensorPrecision CMXGraphFormat::string_to_precision(const char* str) {
    if (!str) return TensorPrecision::UNKNOWN;
    
    for (size_t i = 0; i < PRECISION_NAMES.size(); ++i) {
        if (strcmp(str, PRECISION_NAMES[i]) == 0) {
            return static_cast<TensorPrecision>(i);
        }
    }
    return TensorPrecision::UNKNOWN;
}

TensorAlignment CMXGraphFormat::get_alignment_for_precision(TensorPrecision precision) {
    switch (precision) {
        case TensorPrecision::FLOAT32:
        case TensorPrecision::INT32:
            return TensorAlignment::BYTE_4;
        case TensorPrecision::FLOAT16:
        case TensorPrecision::BFLOAT16:
        case TensorPrecision::INT16:
            return TensorAlignment::BYTE_2;
        case TensorPrecision::INT8:
        case TensorPrecision::UINT8:
        case TensorPrecision::BOOL:
            return TensorAlignment::BYTE_1;
        default:
            return TensorAlignment::BYTE_1;
    }
}

uint32_t CMXGraphFormat::align_offset(uint32_t offset, TensorAlignment alignment) {
    uint32_t align_value = static_cast<uint32_t>(alignment);
    return (offset + align_value - 1) & ~(align_value - 1);
}

bool CMXGraphFormat::is_layout_compatible(DataLayout layout, uint8_t rank) {
    size_t layout_index = static_cast<size_t>(layout);
    if (layout_index >= LAYOUT_RANKS.size()) {
        return false;
    }
    
    uint8_t expected_rank = LAYOUT_RANKS[layout_index];
    return expected_rank == 0 || expected_rank == rank; // 0 means any rank
}

uint8_t CMXGraphFormat::get_expected_rank(DataLayout layout) {
    size_t layout_index = static_cast<size_t>(layout);
    if (layout_index < LAYOUT_RANKS.size()) {
        return LAYOUT_RANKS[layout_index];
    }
    return 0;
}

void CMXGraphFormat::initialize_crc32_table() {
    if (crc32_table_initialized) return;
    
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (uint32_t j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = true;
}

uint32_t CMXGraphFormat::calculate_checksum(const void* data, size_t size) {
    initialize_crc32_table();
    
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < size; ++i) {
        uint8_t table_index = (crc ^ bytes[i]) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[table_index];
    }
    
    return crc ^ 0xFFFFFFFF;
}

void CMXGraphFormat::calculate_strides(const uint32_t* dims, uint8_t rank, DataLayout layout,
                                      uint32_t* strides) {
    if (!dims || !strides || rank == 0) return;
    
    switch (layout) {
        case DataLayout::NHWC:
            if (rank == 4) {
                strides[3] = 1;                           // Channel stride
                strides[2] = dims[3];                     // Width stride
                strides[1] = dims[2] * dims[3];           // Height stride
                strides[0] = dims[1] * dims[2] * dims[3]; // Batch stride
            }
            break;
        case DataLayout::NCHW:
            if (rank == 4) {
                strides[3] = 1;                           // Width stride
                strides[2] = dims[3];                     // Height stride
                strides[1] = dims[2] * dims[3];           // Channel stride
                strides[0] = dims[1] * dims[2] * dims[3]; // Batch stride
            }
            break;
        case DataLayout::HWC:
            if (rank == 3) {
                strides[2] = 1;                           // Channel stride
                strides[1] = dims[2];                     // Width stride
                strides[0] = dims[1] * dims[2];           // Height stride
            }
            break;
        case DataLayout::CHW:
            if (rank == 3) {
                strides[2] = 1;                           // Width stride
                strides[1] = dims[2];                     // Height stride
                strides[0] = dims[1] * dims[2];           // Channel stride
            }
            break;
        case DataLayout::NC:
            if (rank == 2) {
                strides[1] = 1;                           // Channel stride
                strides[0] = dims[1];                     // Batch stride
            }
            break;
        case DataLayout::NHW:
            if (rank == 3) {
                strides[2] = 1;                           // Width stride
                strides[1] = dims[2];                     // Height stride
                strides[0] = dims[1] * dims[2];           // Batch stride
            }
            break;
        case DataLayout::SCALAR:
            if (rank == 1) {
                strides[0] = 1;                           // Scalar stride
            }
            break;
        default:
            // Default row-major ordering
            strides[rank - 1] = 1;
            for (int i = rank - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
            break;
    }
}

const char* CMXGraphFormat::version_to_string(GraphFormatVersion version) {
    switch (version) {
        case GraphFormatVersion::VERSION_1_0:
            return "1.0";
        case GraphFormatVersion::VERSION_1_1:
            return "1.1";
        default:
            return "unknown";
    }
}