#include "cmx_tensor.hpp"
#include <algorithm>
#include <cstring>

namespace cmx {
namespace runtime {

CMXTensor::CMXTensor() 
    : data_(nullptr), dtype_(DataType::FLOAT32), layout_(Layout::NCHW), 
      ndims_(0), owns_data_(false) {
    shape_.fill(0);
    strides_.fill(0);
}

CMXTensor::CMXTensor(void* data, const ShapeArray& shape, DataType dtype, 
                     Layout layout, bool owns_data)
    : data_(data), shape_(shape), dtype_(dtype), layout_(layout), 
      ndims_(0), owns_data_(owns_data) {
    
    // Calculate number of dimensions
    for (size_t i = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (shape_[i] > 0) {
            ndims_ = i + 1;
        }
    }
    
    calculate_strides();
}

CMXTensor::CMXTensor(void* data, std::initializer_list<size_t> shape, 
                     DataType dtype, Layout layout, bool owns_data)
    : data_(data), dtype_(dtype), layout_(layout), ndims_(0), owns_data_(owns_data) {
    
    set_shape_from_list(shape);
    calculate_strides();
}

CMXTensor::~CMXTensor() {
    if (owns_data_ && data_) {
        // Note: In embedded systems, we typically don't use delete/free
        // The allocator should handle cleanup
        data_ = nullptr;
    }
}

CMXTensor::CMXTensor(CMXTensor&& other) noexcept
    : data_(other.data_), shape_(other.shape_), strides_(other.strides_),
      dtype_(other.dtype_), layout_(other.layout_), ndims_(other.ndims_),
      owns_data_(other.owns_data_) {
    
    other.data_ = nullptr;
    other.owns_data_ = false;
    other.ndims_ = 0;
    other.shape_.fill(0);
    other.strides_.fill(0);
}

CMXTensor& CMXTensor::operator=(CMXTensor&& other) noexcept {
    if (this != &other) {
        // Clean up current data if owned
        if (owns_data_ && data_) {
            // Cleanup handled by allocator
            data_ = nullptr;
        }
        
        // Move data from other
        data_ = other.data_;
        shape_ = other.shape_;
        strides_ = other.strides_;
        dtype_ = other.dtype_;
        layout_ = other.layout_;
        ndims_ = other.ndims_;
        owns_data_ = other.owns_data_;
        
        // Reset other
        other.data_ = nullptr;
        other.owns_data_ = false;
        other.ndims_ = 0;
        other.shape_.fill(0);
        other.strides_.fill(0);
    }
    return *this;
}

size_t CMXTensor::size() const {
    size_t total = 1;
    for (size_t i = 0; i < ndims_; ++i) {
        total *= shape_[i];
    }
    return total;
}

size_t CMXTensor::byte_size() const {
    return size() * element_size();
}

size_t CMXTensor::element_size() const {
    return get_element_size(dtype_);
}

bool CMXTensor::empty() const {
    return size() == 0;
}

bool CMXTensor::is_scalar() const {
    return ndims_ == 0 || (ndims_ == 1 && shape_[0] == 1);
}

bool CMXTensor::reshape(const ShapeArray& new_shape) {
    // Calculate total elements in new shape
    size_t new_size = 1;
    size_t new_ndims = 0;
    
    for (size_t i = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (new_shape[i] > 0) {
            new_size *= new_shape[i];
            new_ndims = i + 1;
        }
    }
    
    // Check if total elements match
    if (new_size != size()) {
        return false;
    }
    
    // Apply new shape
    shape_ = new_shape;
    ndims_ = new_ndims;
    calculate_strides();
    
    return true;
}

bool CMXTensor::reshape(std::initializer_list<size_t> new_shape) {
    if (new_shape.size() > MAX_TENSOR_DIMS) {
        return false;
    }
    
    ShapeArray temp_shape;
    temp_shape.fill(0);
    
    size_t i = 0;
    for (auto dim : new_shape) {
        temp_shape[i++] = dim;
    }
    
    return reshape(temp_shape);
}

bool CMXTensor::shape_equals(const CMXTensor& other) const {
    if (ndims_ != other.ndims_) {
        return false;
    }
    
    for (size_t i = 0; i < ndims_; ++i) {
        if (shape_[i] != other.shape_[i]) {
            return false;
        }
    }
    
    return true;
}

bool CMXTensor::is_compatible(const CMXTensor& other) const {
    return shape_equals(other) && dtype_ == other.dtype_;
}

void CMXTensor::reset() {
    if (owns_data_ && data_) {
        // Cleanup handled by allocator
        data_ = nullptr;
    }
    
    data_ = nullptr;
    shape_.fill(0);
    strides_.fill(0);
    ndims_ = 0;
    owns_data_ = false;
}

CMXTensor CMXTensor::wrap(void* data, const ShapeArray& shape, DataType dtype, Layout layout) {
    return CMXTensor(data, shape, dtype, layout, false);
}

CMXTensor CMXTensor::wrap(void* data, std::initializer_list<size_t> shape, 
                         DataType dtype, Layout layout) {
    return CMXTensor(data, shape, dtype, layout, false);
}

CMXTensor CMXTensor::scalar(void* data, DataType dtype) {
    ShapeArray shape;
    shape.fill(0);
    shape[0] = 1;
    return CMXTensor(data, shape, dtype, Layout::SCALAR, false);
}

void CMXTensor::calculate_strides() {
    strides_.fill(0);
    
    if (ndims_ == 0) {
        return;
    }
    
    size_t elem_size = element_size();
    
    switch (layout_) {
        case Layout::NCHW:
            // Row-major: rightmost dimension has stride 1
            if (ndims_ > 0) strides_[ndims_ - 1] = elem_size;
            for (int i = static_cast<int>(ndims_) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
            break;
            
        case Layout::NHWC:
            // Channel-last: channel dimension has stride 1
            if (ndims_ >= 4) {
                strides_[1] = elem_size;  // Channel stride
                strides_[3] = strides_[1] * shape_[1];  // Width stride
                strides_[2] = strides_[3] * shape_[3];  // Height stride
                strides_[0] = strides_[2] * shape_[2];  // Batch stride
            } else {
                // Fall back to row-major for lower dimensions
                if (ndims_ > 0) strides_[ndims_ - 1] = elem_size;
                for (int i = static_cast<int>(ndims_) - 2; i >= 0; --i) {
                    strides_[i] = strides_[i + 1] * shape_[i + 1];
                }
            }
            break;
            
        case Layout::NC:
        case Layout::SCALAR:
        default:
            // Standard row-major layout
            if (ndims_ > 0) strides_[ndims_ - 1] = elem_size;
            for (int i = static_cast<int>(ndims_) - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
            break;
    }
}

size_t CMXTensor::get_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT32:
            return 4;
        case DataType::INT16:
            return 2;
        case DataType::INT8:
            return 1;
        case DataType::UINT8:
            return 1;
        case DataType::BOOL:
            return 1;
        default:
            return 4;  // Default to 4 bytes
    }
}

void CMXTensor::set_shape_from_list(std::initializer_list<size_t> shape_list) {
    shape_.fill(0);
    ndims_ = 0;
    
    if (shape_list.size() > MAX_TENSOR_DIMS) {
        return;  // Invalid shape
    }
    
    size_t i = 0;
    for (auto dim : shape_list) {
        shape_[i++] = dim;
        if (dim > 0) {
            ndims_ = i;
        }
    }
}

} // namespace runtime
} // namespace cmx