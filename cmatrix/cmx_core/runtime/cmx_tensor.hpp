#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <initializer_list>

namespace cmx {
namespace runtime {

/**
 * @brief Enumeration of supported tensor data types
 */
enum class DataType : uint8_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    UINT8 = 5,
    BOOL = 6
};

/**
 * @brief Enumeration of tensor memory layout formats
 */
enum class Layout : uint8_t {
    NCHW = 0,  // Batch, Channel, Height, Width
    NHWC = 1,  // Batch, Height, Width, Channel
    NC = 2,    // Batch, Channel (for 1D/2D tensors)
    SCALAR = 3 // Single value
};

/**
 * @brief Maximum number of tensor dimensions supported
 */
constexpr size_t MAX_TENSOR_DIMS = 4;

/**
 * @brief Lightweight tensor class for embedded AI/ML runtime
 * 
 * This class provides tensor metadata and operations without dynamic memory allocation.
 * It wraps pre-allocated memory and provides shape manipulation, stride calculation,
 * and basic tensor operations suitable for resource-constrained environments.
 */
class CMXTensor {
public:
    using ShapeArray = std::array<size_t, MAX_TENSOR_DIMS>;
    using StrideArray = std::array<size_t, MAX_TENSOR_DIMS>;

    /**
     * @brief Default constructor - creates an empty tensor
     */
    CMXTensor();

    /**
     * @brief Constructor for tensor with specified properties
     * @param data Pointer to tensor data
     * @param shape Tensor dimensions
     * @param dtype Data type of tensor elements
     * @param layout Memory layout format
     * @param owns_data Whether this tensor owns the data pointer
     */
    CMXTensor(void* data, const ShapeArray& shape, DataType dtype, 
              Layout layout = Layout::NCHW, bool owns_data = false);

    /**
     * @brief Constructor with initializer list for shape
     * @param data Pointer to tensor data
     * @param shape Tensor dimensions as initializer list
     * @param dtype Data type of tensor elements
     * @param layout Memory layout format
     * @param owns_data Whether this tensor owns the data pointer
     */
    CMXTensor(void* data, std::initializer_list<size_t> shape, DataType dtype,
              Layout layout = Layout::NCHW, bool owns_data = false);

    /**
     * @brief Destructor - handles data cleanup if owned
     */
    ~CMXTensor();

    // Disable copy constructor and assignment for safety
    CMXTensor(const CMXTensor&) = delete;
    CMXTensor& operator=(const CMXTensor&) = delete;

    // Enable move constructor and assignment
    CMXTensor(CMXTensor&& other) noexcept;
    CMXTensor& operator=(CMXTensor&& other) noexcept;

    /**
     * @brief Get tensor data pointer
     * @return Pointer to tensor data
     */
    void* data() const { return data_; }

    /**
     * @brief Get tensor data pointer as specific type
     * @tparam T Data type to cast to
     * @return Typed pointer to tensor data
     */
    template<typename T>
    T* data() const { return static_cast<T*>(data_); }

    /**
     * @brief Get tensor shape
     * @return Array containing tensor dimensions
     */
    const ShapeArray& shape() const { return shape_; }

    /**
     * @brief Get tensor strides
     * @return Array containing tensor strides
     */
    const StrideArray& strides() const { return strides_; }

    /**
     * @brief Get tensor data type
     * @return Data type enumeration
     */
    DataType dtype() const { return dtype_; }

    /**
     * @brief Get tensor layout
     * @return Layout enumeration
     */
    Layout layout() const { return layout_; }

    /**
     * @brief Get number of dimensions
     * @return Number of tensor dimensions
     */
    size_t ndims() const { return ndims_; }

    /**
     * @brief Check if tensor owns its data
     * @return True if tensor owns data, false otherwise
     */
    bool owns_data() const { return owns_data_; }

    /**
     * @brief Calculate total number of elements
     * @return Total element count
     */
    size_t size() const;

    /**
     * @brief Calculate total size in bytes
     * @return Total tensor size in bytes
     */
    size_t byte_size() const;

    /**
     * @brief Get size of a single element in bytes
     * @return Element size in bytes
     */
    size_t element_size() const;

    /**
     * @brief Check if tensor is empty
     * @return True if tensor has no elements
     */
    bool empty() const;

    /**
     * @brief Check if tensor is scalar (0-dimensional)
     * @return True if tensor is scalar
     */
    bool is_scalar() const;

    /**
     * @brief Reshape tensor (must maintain total element count)
     * @param new_shape New tensor shape
     * @return True if reshape was successful
     */
    bool reshape(const ShapeArray& new_shape);

    /**
     * @brief Reshape tensor with initializer list
     * @param new_shape New tensor shape
     * @return True if reshape was successful
     */
    bool reshape(std::initializer_list<size_t> new_shape);

    /**
     * @brief Compare tensor shapes
     * @param other Tensor to compare with
     * @return True if shapes are identical
     */
    bool shape_equals(const CMXTensor& other) const;

    /**
     * @brief Check if tensors are compatible for element-wise operations
     * @param other Tensor to check compatibility with
     * @return True if tensors are compatible
     */
    bool is_compatible(const CMXTensor& other) const;

    /**
     * @brief Reset tensor to empty state
     */
    void reset();

    // Static factory methods

    /**
     * @brief Create a tensor wrapping existing data
     * @param data Pointer to existing data
     * @param shape Tensor dimensions
     * @param dtype Data type
     * @param layout Memory layout
     * @return CMXTensor wrapping the data
     */
    static CMXTensor wrap(void* data, const ShapeArray& shape, DataType dtype,
                         Layout layout = Layout::NCHW);

    /**
     * @brief Create a tensor wrapping existing data with initializer list shape
     * @param data Pointer to existing data
     * @param shape Tensor dimensions
     * @param dtype Data type
     * @param layout Memory layout
     * @return CMXTensor wrapping the data
     */
    static CMXTensor wrap(void* data, std::initializer_list<size_t> shape, 
                         DataType dtype, Layout layout = Layout::NCHW);

    /**
     * @brief Create a scalar tensor
     * @param data Pointer to scalar value
     * @param dtype Data type
     * @return Scalar CMXTensor
     */
    static CMXTensor scalar(void* data, DataType dtype);

private:
    void* data_;                    ///< Pointer to tensor data
    ShapeArray shape_;              ///< Tensor dimensions
    StrideArray strides_;           ///< Memory strides for each dimension
    DataType dtype_;                ///< Data type of tensor elements
    Layout layout_;                 ///< Memory layout format
    size_t ndims_;                  ///< Number of dimensions
    bool owns_data_;                ///< Whether tensor owns the data pointer

    /**
     * @brief Calculate strides based on shape and layout
     */
    void calculate_strides();

    /**
     * @brief Get element size for given data type
     * @param dtype Data type
     * @return Size in bytes
     */
    static size_t get_element_size(DataType dtype);

    /**
     * @brief Set shape from initializer list
     * @param shape_list Initializer list of dimensions
     */
    void set_shape_from_list(std::initializer_list<size_t> shape_list);
};

} // namespace runtime
} // namespace cmx