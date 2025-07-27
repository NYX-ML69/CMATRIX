#include "cmx_node.hpp"
#include <sstream>
#include <algorithm>

namespace cmx {
namespace graph {

CMXNode::CMXNode(CMXOpType op_type, const std::string& name)
    : op_type_(op_type), name_(name), status_(CMXNodeStatus::READY) {
    if (name_.empty()) {
        name_ = op_utils::op_type_to_string(op_type_);
    }
}

CMXNode::CMXNode(const CMXNode& other)
    : op_type_(other.op_type_), name_(other.name_), inputs_(other.inputs_),
      outputs_(other.outputs_), attributes_(other.attributes_), status_(other.status_) {}

CMXNode& CMXNode::operator=(const CMXNode& other) {
    if (this != &other) {
        op_type_ = other.op_type_;
        name_ = other.name_;
        inputs_ = other.inputs_;
        outputs_ = other.outputs_;
        attributes_ = other.attributes_;
        status_ = other.status_;
    }
    return *this;
}

CMXNode::~CMXNode() {
    clear_inputs();
    clear_outputs();
    clear_attributes();
}

CMXOpType CMXNode::get_op_type() const {
    return op_type_;
}

const std::string& CMXNode::get_name() const {
    return name_;
}

void CMXNode::set_name(const std::string& name) {
    name_ = name;
}

void CMXNode::add_input(TensorID tensor_id) {
    if (tensor_id != 0) {
        inputs_.push_back(tensor_id);
    }
}

void CMXNode::add_output(TensorID tensor_id) {
    if (tensor_id != 0) {
        outputs_.push_back(tensor_id);
    }
}

const std::vector<TensorID>& CMXNode::get_inputs() const {
    return inputs_;
}

const std::vector<TensorID>& CMXNode::get_outputs() const {
    return outputs_;
}

TensorID CMXNode::get_input(size_t index) const {
    return (index < inputs_.size()) ? inputs_[index] : 0;
}

TensorID CMXNode::get_output(size_t index) const {
    return (index < outputs_.size()) ? outputs_[index] : 0;
}

size_t CMXNode::input_count() const {
    return inputs_.size();
}

size_t CMXNode::output_count() const {
    return outputs_.size();
}

void CMXNode::clear_inputs() {
    inputs_.clear();
}

void CMXNode::clear_outputs() {
    outputs_.clear();
}

void CMXNode::set_attribute(const std::string& key, const AttributeValue& value) {
    attributes_[key] = value;
}

const AttributeValue* CMXNode::get_attribute(const std::string& key) const {
    auto it = attributes_.find(key);
    return (it != attributes_.end()) ? &it->second : nullptr;
}

bool CMXNode::has_attribute(const std::string& key) const {
    return attributes_.find(key) != attributes_.end();
}

std::vector<std::string> CMXNode::get_attribute_keys() const {
    std::vector<std::string> keys;
    for (const auto& [key, _] : attributes_) {
        keys.push_back(key);
    }
    return keys;
}

bool CMXNode::remove_attribute(const std::string& key) {
    return attributes_.erase(key) > 0;
}

void CMXNode::clear_attributes() {
    attributes_.clear();
}

void CMXNode::set_int_attribute(const std::string& key, int32_t value) {
    attributes_[key] = value;
}

void CMXNode::set_float_attribute(const std::string& key, float value) {
    attributes_[key] = value;
}

void CMXNode::set_string_attribute(const std::string& key, const std::string& value) {
    attributes_[key] = value;
}

int32_t CMXNode::get_int_attribute(const std::string& key, int32_t default_value) const {
    auto it = attributes_.find(key);
    if (it != attributes_.end()) {
        if (std::holds_alternative<int32_t>(it->second)) {
            return std::get<int32_t>(it->second);
        }
    }
    return default_value;
}

float CMXNode::get_float_attribute(const std::string& key, float default_value) const {
    auto it = attributes_.find(key);
    if (it != attributes_.end()) {
        if (std::holds_alternative<float>(it->second)) {
            return std::get<float>(it->second);
        }
    }
    return default_value;
}

std::string CMXNode::get_string_attribute(const std::string& key, const std::string& default_value) const {
    auto it = attributes_.find(key);
    if (it != attributes_.end()) {
        if (std::holds_alternative<std::string>(it->second)) {
            return std::get<std::string>(it->second);
        }
    }
    return default_value;
}

bool CMXNode::execute(const std::vector<void*>& input_tensors, 
                     const std::vector<void*>& output_tensors) const {
    // Placeholder for execution logic
    // In a real implementation, this would dispatch to operation-specific handlers
    status_ = CMXNodeStatus::RUNNING;
    
    // Basic validation
    if (input_tensors.size() != inputs_.size() || output_tensors.size() != outputs_.size()) {
        return false;
    }
    
    // TODO: Implement actual operation execution based on op_type_
    // This would involve calling specific kernels or operation implementations
    
    return true;
}

CMXNodeStatus CMXNode::get_status() const {
    return status_;
}

void CMXNode::set_status(CMXNodeStatus status) {
    status_ = status;
}

bool CMXNode::validate() const {
    return validate_inputs() && validate_outputs() && validate_attributes();
}

std::string CMXNode::to_string() const {
    std::ostringstream oss;
    oss << "CMXNode {";
    oss << " op_type: " << get_op_type_string();
    oss << ", name: \"" << name_ << "\"";
    oss << ", inputs: [";
    for (size_t i = 0; i < inputs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << inputs_[i];
    }
    oss << "], outputs: [";
    for (size_t i = 0; i < outputs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << outputs_[i];
    }
    oss << "], attributes: {";
    
    bool first = true;
    for (const auto& [key, value] : attributes_) {
        if (!first) oss << ", ";
        oss << key << ": " << format_attribute_value(value);
        first = false;
    }
    oss << "} }";
    
    return oss.str();
}

std::string CMXNode::get_op_type_string() const {
    return op_utils::op_type_to_string(op_type_);
}

size_t CMXNode::get_memory_usage() const {
    // Basic memory usage estimation
    size_t usage = sizeof(CMXNode);
    usage += name_.capacity();
    usage += inputs_.capacity() * sizeof(TensorID);
    usage += outputs_.capacity() * sizeof(TensorID);
    
    // Estimate attribute memory usage
    for (const auto& [key, value] : attributes_) {
        usage += key.capacity();
        usage += std::visit([](const auto& v) -> size_t {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
                return v.capacity();
            } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                return v.capacity() * sizeof(int32_t);
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return v.capacity() * sizeof(float);
            } else {
                return sizeof(T);
            }
        }, value);
    }
    
    return usage;
}

uint64_t CMXNode::get_computation_cost() const {
    // Basic computation cost estimation based on operation type
    // This is a simplified heuristic and would need to be refined based on actual profiling
    switch (op_type_) {
        case CMXOpType::ADD:
        case CMXOpType::SUB:
        case CMXOpType::MUL:
        case CMXOpType::DIV:
            return 1;
        
        case CMXOpType::RELU:
        case CMXOpType::SIGMOID:
        case CMXOpType::TANH:
            return 2;
        
        case CMXOpType::CONV2D:
        case CMXOpType::DEPTHWISE_CONV2D:
            return 100;
        
        case CMXOpType::MATMUL:
        case CMXOpType::BATCH_MATMUL:
            return 50;
        
        case CMXOpType::BATCH_NORM:
        case CMXOpType::LAYER_NORM:
            return 10;
        
        default:
            return 5;
    }
}

bool CMXNode::validate_inputs() const {
    // Check that all input tensor IDs are valid (non-zero)
    for (TensorID input : inputs_) {
        if (input == 0) {
            return false;
        }
    }
    return true;
}

bool CMXNode::validate_outputs() const {
    // Check that all output tensor IDs are valid (non-zero)
    for (TensorID output : outputs_) {
        if (output == 0) {
            return false;
        }
    }
    return true;
}

bool CMXNode::validate_attributes() const {
    // Basic attribute validation
    // This could be extended with operation-specific validation
    return true;
}

std::string CMXNode::format_attribute_value(const AttributeValue& value) const {
    return std::visit([](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, int32_t>) {
            return std::to_string(v);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::to_string(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "\"" + v + "\"";
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
            std::string result = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(v[i]);
            }
            result += "]";
            return result;
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            std::string result = "[";
            for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) result += ", ";
                result += std::to_string(v[i]);
            }
            result += "]";
            return result;
        }
        return "unknown";
    }, value);
}

namespace op_utils {

std::string op_type_to_string(CMXOpType op_type) {
    switch (op_type) {
        case CMXOpType::UNKNOWN: return "UNKNOWN";
        case CMXOpType::ADD: return "ADD";
        case CMXOpType::SUB: return "SUB";
        case CMXOpType::MUL: return "MUL";
        case CMXOpType::DIV: return "DIV";
        case CMXOpType::RELU: return "RELU";
        case CMXOpType::SIGMOID: return "SIGMOID";
        case CMXOpType::TANH: return "TANH";
        case CMXOpType::SOFTMAX: return "SOFTMAX";
        case CMXOpType::CONV2D: return "CONV2D";
        case CMXOpType::DEPTHWISE_CONV2D: return "DEPTHWISE_CONV2D";
        case CMXOpType::TRANSPOSE_CONV2D: return "TRANSPOSE_CONV2D";
        case CMXOpType::MAX_POOL2D: return "MAX_POOL2D";
        case CMXOpType::AVG_POOL2D: return "AVG_POOL2D";
        case CMXOpType::GLOBAL_AVG_POOL2D: return "GLOBAL_AVG_POOL2D";
        case CMXOpType::MATMUL: return "MATMUL";
        case CMXOpType::BATCH_MATMUL: return "BATCH_MATMUL";
        case CMXOpType::BATCH_NORM: return "BATCH_NORM";
        case CMXOpType::LAYER_NORM: return "LAYER_NORM";
        case CMXOpType::INSTANCE_NORM: return "INSTANCE_NORM";
        case CMXOpType::RESHAPE: return "RESHAPE";
        case CMXOpType::TRANSPOSE: return "TRANSPOSE";
        case CMXOpType::SQUEEZE: return "SQUEEZE";
        case CMXOpType::UNSQUEEZE: return "UNSQUEEZE";
        case CMXOpType::REDUCE_MEAN: return "REDUCE_MEAN";
        case CMXOpType::REDUCE_SUM: return "REDUCE_SUM";
        case CMXOpType::REDUCE_MAX: return "REDUCE_MAX";
        case CMXOpType::REDUCE_MIN: return "REDUCE_MIN";
        case CMXOpType::CONCAT: return "CONCAT";
        case CMXOpType::SPLIT: return "SPLIT";
        case CMXOpType::PAD: return "PAD";
        case CMXOpType::SLICE: return "SLICE";
        case CMXOpType::CUSTOM: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

CMXOpType string_to_op_type(const std::string& op_string) {
    if (op_string == "ADD") return CMXOpType::ADD;
    if (op_string == "SUB") return CMXOpType::SUB;
    if (op_string == "MUL") return CMXOpType::MUL;
    if (op_string == "DIV") return CMXOpType::DIV;
    if (op_string == "RELU") return CMXOpType::RELU;
    if (op_string == "SIGMOID") return CMXOpType::SIGMOID;
    if (op_string == "TANH") return CMXOpType::TANH;
    if (op_string == "SOFTMAX") return CMXOpType::SOFTMAX;
    if (op_string == "CONV2D") return CMXOpType::CONV2D;
    if (op_string == "DEPTHWISE_CONV2D") return CMXOpType::DEPTHWISE_CONV2D;
    if (op_string == "TRANSPOSE_CONV2D") return CMXOpType::TRANSPOSE_CONV2D;
    if (op_string == "MAX_POOL2D") return CMXOpType::MAX_POOL2D;
    if (op_string == "AVG_POOL2D") return CMXOpType::AVG_POOL2D;
    if (op_string == "GLOBAL_AVG_POOL2D") return CMXOpType::GLOBAL_AVG_POOL2D;
    if (op_string == "MATMUL") return CMXOpType::MATMUL;
    if (op_string == "BATCH_MATMUL") return CMXOpType::BATCH_MATMUL;
    if (op_string == "BATCH_NORM") return CMXOpType::BATCH_NORM;
    if (op_string == "LAYER_NORM") return CMXOpType::LAYER_NORM;
    if (op_string == "INSTANCE_NORM") return CMXOpType::INSTANCE_NORM;
    if (op_string == "RESHAPE") return CMXOpType::RESHAPE;
    if (op_string == "TRANSPOSE") return CMXOpType::TRANSPOSE;
    if (op_string == "SQUEEZE") return CMXOpType::SQUEEZE;
    if (op_string == "UNSQUEEZE") return CMXOpType::UNSQUEEZE;
    if (op_string == "REDUCE_MEAN") return CMXOpType::REDUCE_MEAN;
    if (op_string == "REDUCE_SUM") return CMXOpType::REDUCE_SUM;
    if (op_string == "REDUCE_MAX") return CMXOpType::REDUCE_MAX;
    if (op_string == "REDUCE_MIN") return CMXOpType::REDUCE_MIN;
    if (op_string == "CONCAT") return CMXOpType::CONCAT;
    if (op_string == "SPLIT") return CMXOpType::SPLIT;
    if (op_string == "PAD") return CMXOpType::PAD;
    if (op_string == "SLICE") return CMXOpType::SLICE;
    if (op_string == "CUSTOM") return CMXOpType::CUSTOM;
    return CMXOpType::UNKNOWN;
}

bool is_supported_op_type(CMXOpType op_type) {
    return op_type != CMXOpType::UNKNOWN;
}

} // namespace op_utils

} // namespace graph
} // namespace cmx