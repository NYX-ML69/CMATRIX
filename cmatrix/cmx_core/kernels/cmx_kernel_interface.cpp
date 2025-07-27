#include "cmx_kernel_interface.hpp"

namespace cmx {
namespace kernels {

int32_t TensorDescriptor::size_bytes() const {
    int32_t element_size = 0;
    
    switch (dtype) {
        case DataType::FLOAT32:
            element_size = 4;
            break;
        case DataType::INT32:
            element_size = 4;
            break;
        case DataType::INT16:
            element_size = 2;
            break;
        case DataType::INT8:
        case DataType::UINT8:
            element_size = 1;
            break;
        default:
            return 0;
    }
    
    return shape.total_size() * element_size;
}

} // namespace kernels
} // namespace cmx