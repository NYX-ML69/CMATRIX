#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <variant>

namespace cmx {
namespace graph {

using TensorID = uint32_t;
using AttributeValue = std::variant<int32_t, float, std::string, std::vector<int32_t>, std::vector<float>>;

/**
 * @brief Operation type enumeration
 */
enum class CMXOpType : uint16_t {
    UNKNOWN = 0,
    
    // Basic operations
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    
    // Activation functions
    RELU = 10,
    SIGMOID = 11,
    TANH = 12,
    SOFTMAX = 13,
    
    // Convolution operations
    CONV2D = 20,
    DEPTHWISE_CONV2D = 21,
    TRANSPOSE_CONV2D = 22,
    
    // Pooling operations
    MAX_POOL2D = 30,
    AVG_POOL2D = 31,
    GLOBAL_AVG_POOL2D = 32,
    
    // Matrix operations
    MATMUL = 40,
    BATCH_MATMUL = 41,
    
    // Normalization
    BATCH_NORM = 50,
    LAYER_NORM = 51,
    INSTANCE_NORM = 52,
    
    // Reshape operations
    RESHAPE = 60,
    TRANSPOSE = 61,
    SQUEEZE = 62,
    UNSQUEEZE = 63,
    
    // Reduction operations
    REDUCE_MEAN = 70,
    REDUCE_SUM = 71,
    REDUCE_MAX = 72,
    REDUCE_MIN = 73,
    
    // Utility operations
    CONCAT = 80,
    SPLIT = 81,
    PAD = 82,
    SLICE = 83,
    
    // Custom operations
    CUSTOM = 1000
};

/**
 * @brief Node execution status
 */
enum class CMXNodeStatus : uint8_t {
    READY = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3
};

/**
 * @brief Computation node in the graph
 * 
 * Represents a single operation with inputs, outputs, and attributes.
 * Used as building blocks for computation graphs.
 */
class CMXNode {
public:
    /**
     * @brief Constructor
     * @param op_type Type of operation this node performs
     * @param name Optional name for the node
     */
    explicit CMXNode(CMXOpType op_type, const std::string& name = "");
    
    /**
     * @brief Copy constructor
     */
    CMXNode(const CMXNode& other);
    
    /**
     * @brief Assignment operator
     */
    CMXNode& operator=(const CMXNode& other);
    
    /**
     * @brief Destructor
     */
    ~CMXNode();

    // Basic properties
    /**
     * @brief Get the operation type
     * @return Operation type enum
     */
    CMXOpType get_op_type() const;
    
    /**
     * @brief Get the node name
     * @return Node name string
     */
    const std::string& get_name() const;
    
    /**
     * @brief Set the node name
     * @param name New name for the node
     */
    void set_name(const std::string& name);

    // Input/Output management
    /**
     * @brief Add input tensor ID
     * @param tensor_id ID of input tensor
     */
    void add_input(TensorID tensor_id);
    
    /**
     * @brief Add output tensor ID
     * @param tensor_id ID of output tensor
     */
    void add_output(TensorID tensor_id);
    
    /**
     * @brief Get all input tensor IDs
     * @return Vector of input tensor IDs
     */
    const std::vector<TensorID>& get_inputs() const;
    
    /**
     * @brief Get all output tensor IDs
     * @return Vector of output tensor IDs
     */
    const std::vector<TensorID>& get_outputs() const;
    
    /**
     * @brief Get input tensor ID at specific index
     * @param index Input index
     * @return Tensor ID, or 0 if index out of bounds
     */
    TensorID get_input(size_t index) const;
    
    /**
     * @brief Get output tensor ID at specific index
     * @param index Output index
     * @return Tensor ID, or 0 if index out of bounds
     */
    TensorID get_output(size_t index) const;
    
    /**
     * @brief Get number of inputs
     * @return Number of input tensors
     */
    size_t input_count() const;
    
    /**
     * @brief Get number of outputs
     * @return Number of output tensors
     */
    size_t output_count() const;
    
    /**
     * @brief Clear all inputs
     */
    void clear_inputs();
    
    /**
     * @brief Clear all outputs
     */
    void clear_outputs();

    // Attribute management
    /**
     * @brief Set an attribute value
     * @param key Attribute name
     * @param value Attribute value
     */
    void set_attribute(const std::string& key, const AttributeValue& value);
    
    /**
     * @brief Get attribute value
     * @param key Attribute name
     * @return Pointer to attribute value, nullptr if not found
     */
    const AttributeValue* get_attribute(const std::string& key) const;
    
    /**
     * @brief Check if attribute exists
     * @param key Attribute name
     * @return true if attribute exists, false otherwise
     */
    bool has_attribute(const std::string& key) const;
    
    /**
     * @brief Get all attribute keys
     * @return Vector of attribute keys
     */
    std::vector<std::string> get_attribute_keys() const;
    
    /**
     * @brief Remove an attribute
     * @param key Attribute name
     * @return true if attribute was removed, false if not found
     */
    bool remove_attribute(const std::string& key);
    
    /**
     * @brief Clear all attributes
     */
    void clear_attributes();

    // Utility methods for common attributes
    /**
     * @brief Set integer attribute
     * @param key Attribute name
     * @param value Integer value
     */
    void set_int_attribute(const std::string& key, int32_t value);
    
    /**
     * @brief Set float attribute
     * @param key Attribute name
     * @param value Float value
     */
    void set_float_attribute(const std::string& key, float value);
    
    /**
     * @brief Set string attribute
     * @param key Attribute name
     * @param value String value
     */
    void set_string_attribute(const std::string& key, const std::string& value);
    
    /**
     * @brief Get integer attribute
     * @param key Attribute name
     * @param default_value Default value if attribute not found
     * @return Integer value or default
     */
    int32_t get_int_attribute(const std::string& key, int32_t default_value = 0) const;
    
    /**
     * @brief Get float attribute
     * @param key Attribute name
     * @param default_value Default value if attribute not found
     * @return Float value or default
     */
    float get_float_attribute(const std::string& key, float default_value = 0.0f) const;
    
    /**
     * @brief Get string attribute
     * @param key Attribute name
     * @param default_value Default value if attribute not found
     * @return String value or default
     */
    std::string get_string_attribute(const std::string& key, const std::string& default_value = "") const;

    // Execution interface (for interpreted fallback)
    /**
     * @brief Execute the node operation
     * @param input_tensors Vector of input tensor pointers
     * @param output_tensors Vector of output tensor pointers
     * @return true if execution successful, false otherwise
     */
    bool execute(const std::vector<void*>& input_tensors, 
                 const std::vector<void*>& output_tensors) const;
    
    /**
     * @brief Get current execution status
     * @return Node execution status
     */
    CMXNodeStatus get_status() const;
    
    /**
     * @brief Set execution status
     * @param status New execution status
     */
    void set_status(CMXNodeStatus status);

    // Validation and debugging
    /**
     * @brief Validate node configuration
     * @return true if node is valid, false otherwise
     */
    bool validate() const;
    
    /**
     * @brief Generate string representation of the node
     * @return String describing the node
     */
    std::string to_string() const;
    
    /**
     * @brief Get operation type as string
     * @return String representation of operation type
     */
    std::string get_op_type_string() const;

    // Memory and performance
    /**
     * @brief Get estimated memory usage for this node
     * @return Memory usage in bytes
     */
    size_t get_memory_usage() const;
    
    /**
     * @brief Get estimated computation cost
     * @return Relative computation cost (arbitrary units)
     */
    uint64_t get_computation_cost() const;

private:
    CMXOpType op_type_;
    std::string name_;
    std::vector<TensorID> inputs_;
    std::vector<TensorID> outputs_;
    std::unordered_map<std::string, AttributeValue> attributes_;
    CMXNodeStatus status_;

    // Internal helper methods
    bool validate_inputs() const;
    bool validate_outputs() const;
    bool validate_attributes() const;
    std::string format_attribute_value(const AttributeValue& value) const;
};

/**
 * @brief Utility functions for operation types
 */
namespace op_utils {
    /**
     * @brief Convert operation type to string
     * @param op_type Operation type enum
     * @return String representation
     */
    std::string op_type_to_string(CMXOpType op_type);
    
    /**
     * @brief Convert string to operation type
     * @param op_string String representation
     * @return Operation type enum
     */
    CMXOpType string_to_op_type(const std::string& op_string);
    
    /**
     * @brief Check if operation type is supported
     * @param op_type Operation type enum
     * @return true if supported, false otherwise
     */
    bool is_supported_op_type(CMXOpType op_type);
}

} // namespace graph
} // namespace cmx