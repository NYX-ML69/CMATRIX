#include "cmx_op_loader.hpp"
#include "../core/cmx_status.hpp"
#include "../core/cmx_tensor.hpp"
#include "cmx_ops.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

namespace cmx {

// Magic bytes for format detection
static const uint8_t CMX_BINARY_MAGIC[] = {'C', 'M', 'X', 'B'};
static const uint8_t ONNX_MAGIC[] = {0x08, 0x01, 0x12};  // Simplified ONNX signature
static const uint8_t TFLITE_MAGIC[] = {'T', 'F', 'L', '3'};

cmx_status cmx_op_loader::load_from_file(
    const char* file_path,
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    if (!file_path) {
        return cmx_status::INVALID_ARGUMENT;
    }
    
    // Open and read file
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return cmx_status::FILE_NOT_FOUND;
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read entire file into buffer
    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);
    file.close();
    
    // Detect format and load
    cmx_model_format format = detect_format(file_path, buffer.data(), file_size);
    return load_from_buffer(buffer.data(), file_size, format, model, config);
}

cmx_status cmx_op_loader::load_from_buffer(
    const uint8_t* buffer,
    size_t buffer_size,
    cmx_model_format format,
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    if (!buffer || buffer_size == 0) {
        return cmx_status::INVALID_ARGUMENT;
    }
    
    // Clear previous model data
    free_model(model);
    model.format = format;
    
    cmx_status status = cmx_status::SUCCESS;
    
    switch (format) {
        case cmx_model_format::CMX_BINARY:
            status = load_cmx_binary(buffer, buffer_size, model, config);
            break;
            
        case cmx_model_format::CMX_TEXT:
            status = load_cmx_text(reinterpret_cast<const char*>(buffer), model, config);
            break;
            
        case cmx_model_format::ONNX:
        case cmx_model_format::TFLITE:
            // External format support would go here
            return cmx_status::NOT_IMPLEMENTED;
            
        default:
            return cmx_status::INVALID_FORMAT;
    }
    
    if (status != cmx_status::SUCCESS) {
        return status;
    }
    
    // Post-processing steps
    if (config.enable_shape_inference) {
        status = infer_shapes(model, config);
        if (status != cmx_status::SUCCESS) return status;
    }
    
    if (config.enable_layout_optimization) {
        status = optimize_layouts(model, config);
        if (status != cmx_status::SUCCESS) return status;
    }
    
    if (config.enable_constant_folding) {
        status = fold_constants(model, config);
        if (status != cmx_status::SUCCESS) return status;
    }
    
    // Final validation
    return validate_model(model);
}

cmx_status cmx_op_loader::load_from_string(
    const char* model_string,
    cmx_model_format format,
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    if (!model_string) {
        return cmx_status::INVALID_ARGUMENT;
    }
    
    size_t string_len = strlen(model_string);
    return load_from_buffer(
        reinterpret_cast<const uint8_t*>(model_string),
        string_len,
        format,
        model,
        config
    );
}

cmx_status cmx_op_loader::create_executable_graph(
    const cmx_loaded_model& model,
    cmx_op_graph& graph
) {
    // Convert loaded model to executable format
    // This would integrate with cmx_graph_executor
    
    // Sort nodes in execution order
    std::vector<cmx_op_node> sorted_nodes = model.nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
        [](const cmx_op_node& a, const cmx_op_node& b) {
            return a.execution_order < b.execution_order;
        });
    
    // Create graph structure (implementation depends on cmx_op_graph definition)
    // graph.nodes = sorted_nodes;
    // graph.tensors = model.tensors;
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::bind_tensors(
    const cmx_loaded_model& model,
    cmx_tensor* input_tensors,
    size_t num_inputs,
    cmx_tensor* output_tensors,
    size_t num_outputs
) {
    if (!input_tensors || !output_tensors) {
        return cmx_status::INVALID_ARGUMENT;
    }
    
    size_t input_count = 0, output_count = 0;
    
    // Count actual inputs/outputs in model
    for (const auto& binding : model.tensors) {
        if (binding.is_input) input_count++;
        if (binding.is_output) output_count++;
    }
    
    if (input_count != num_inputs || output_count != num_outputs) {
        return cmx_status::SHAPE_MISMATCH;
    }
    
    // Bind input tensors
    size_t input_idx = 0;
    for (const auto& binding : model.tensors) {
        if (binding.is_input && input_idx < num_inputs) {
            // Validate shape compatibility
            if (input_tensors[input_idx].rank != binding.shape.size()) {
                return cmx_status::SHAPE_MISMATCH;
            }
            
            for (size_t i = 0; i < binding.shape.size(); ++i) {
                if (input_tensors[input_idx].shape[i] != binding.shape[i]) {
                    return cmx_status::SHAPE_MISMATCH;
                }
            }
            
            input_idx++;
        }
    }
    
    // Bind output tensors similarly
    size_t output_idx = 0;
    for (const auto& binding : model.tensors) {
        if (binding.is_output && output_idx < num_outputs) {
            // Set output tensor properties from model
            output_tensors[output_idx].dtype = binding.dtype;
            output_tensors[output_idx].rank = binding.shape.size();
            
            for (size_t i = 0; i < binding.shape.size(); ++i) {
                output_tensors[output_idx].shape[i] = binding.shape[i];
            }
            
            output_idx++;
        }
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::validate_model(const cmx_loaded_model& model) {
    // Check for empty model
    if (model.nodes.empty()) {
        return cmx_status::INVALID_MODEL;
    }
    
    // Validate tensor connections
    cmx_status status = validate_connections(model);
    if (status != cmx_status::SUCCESS) {
        return status;
    }
    
    // Check for at least one input and output
    bool has_input = false, has_output = false;
    for (const auto& tensor : model.tensors) {
        if (tensor.is_input) has_input = true;
        if (tensor.is_output) has_output = true;
    }
    
    if (!has_input || !has_output) {
        return cmx_status::INVALID_MODEL;
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::get_model_info(
    const cmx_loaded_model& model,
    size_t& num_inputs,
    size_t& num_outputs,
    size_t& num_operations
) {
    num_inputs = 0;
    num_outputs = 0;
    num_operations = model.nodes.size();
    
    for (const auto& tensor : model.tensors) {
        if (tensor.is_input) num_inputs++;
        if (tensor.is_output) num_outputs++;
    }
    
    return cmx_status::SUCCESS;
}

void cmx_op_loader::free_model(cmx_loaded_model& model) {
    model.nodes.clear();
    model.tensors.clear();
    model.constant_data.clear();
    model.model_name.clear();
    model.model_version = 0;
}

cmx_model_format cmx_op_loader::detect_format(
    const char* file_path,
    const uint8_t* buffer,
    size_t buffer_size
) {
    // Check file extension first
    if (file_path) {
        std::string path(file_path);
        std::string ext = path.substr(path.find_last_of('.'));
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".cmxb") return cmx_model_format::CMX_BINARY;
        if (ext == ".cmxt") return cmx_model_format::CMX_TEXT;
        if (ext == ".onnx") return cmx_model_format::ONNX;
        if (ext == ".tflite") return cmx_model_format::TFLITE;
    }
    
    // Check magic bytes if buffer provided
    if (buffer && buffer_size >= 4) {
        if (cmx_format_detector::is_cmx_binary(buffer, buffer_size)) {
            return cmx_model_format::CMX_BINARY;
        }
        if (cmx_format_detector::is_onnx(buffer, buffer_size)) {
            return cmx_model_format::ONNX;
        }
        if (cmx_format_detector::is_tflite(buffer, buffer_size)) {
            return cmx_model_format::TFLITE;
        }
        if (cmx_format_detector::is_cmx_text(reinterpret_cast<const char*>(buffer))) {
            return cmx_model_format::CMX_TEXT;
        }
    }
    
    return cmx_model_format::CMX_BINARY; // Default fallback
}

cmx_status cmx_op_loader::load_cmx_binary(
    const uint8_t* buffer,
    size_t buffer_size,
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    if (buffer_size < sizeof(CMX_BINARY_MAGIC) + 8) {
        return cmx_status::INVALID_FORMAT;
    }
    
    const uint8_t* ptr = buffer;
    
    // Check magic bytes
    if (memcmp(ptr, CMX_BINARY_MAGIC, sizeof(CMX_BINARY_MAGIC)) != 0) {
        return cmx_status::INVALID_FORMAT;
    }
    ptr += sizeof(CMX_BINARY_MAGIC);
    
    // Read version
    model.model_version = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    
    // Read number of nodes
    uint32_t num_nodes = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    
    if (num_nodes > config.max_graph_nodes) {
        return cmx_status::RESOURCE_EXHAUSTED;
    }
    
    // Read nodes
    model.nodes.reserve(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        cmx_op_node node;
        
        // Read node name length and data
        uint16_t name_len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);
        
        node.node_name.assign(reinterpret_cast<const char*>(ptr), name_len);
        ptr += name_len;
        
        // Read op type length and data
        uint16_t op_type_len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);
        
        node.op_type.assign(reinterpret_cast<const char*>(ptr), op_type_len);
        ptr += op_type_len;
        
        // Read input/output counts
        uint16_t num_inputs = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);
        
        uint16_t num_outputs = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);
        
        // Read input IDs
        node.input_ids.resize(num_inputs);
        for (uint16_t j = 0; j < num_inputs; ++j) {
            node.input_ids[j] = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += sizeof(uint32_t);
        }
        
        // Read output IDs
        node.output_ids.resize(num_outputs);
        for (uint16_t j = 0; j < num_outputs; ++j) {
            node.output_ids[j] = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += sizeof(uint32_t);
        }
        
        // Read execution order
        node.execution_order = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        
        // Read op parameters
        uint32_t param_size = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        
        if (param_size > 0) {
            node.op_params.assign(ptr, ptr + param_size);
            ptr += param_size;
        }
        
        model.nodes.push_back(std::move(node));
    }
    
    // Read tensor bindings
    uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    
    if (num_tensors > config.max_tensor_count) {
        return cmx_status::RESOURCE_EXHAUSTED;
    }
    
    model.tensors.reserve(num_tensors);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        cmx_tensor_binding binding;
        
        // Read tensor name
        uint16_t name_len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += sizeof(uint16_t);
        
        binding.name.assign(reinterpret_cast<const char*>(ptr), name_len);
        ptr += name_len;
        
        // Read tensor properties
        binding.tensor_id = *reinterpret_cast<const uint32_t*>(ptr);
        ptr += sizeof(uint32_t);
        
        binding.dtype = static_cast<cmx_tensor_dtype>(*ptr++);
        
        uint8_t rank = *ptr++;
        binding.shape.resize(rank);
        for (uint8_t j = 0; j < rank; ++j) {
            binding.shape[j] = *reinterpret_cast<const uint32_t*>(ptr);
            ptr += sizeof(uint32_t);
        }
        
        binding.byte_offset = *reinterpret_cast<const size_t*>(ptr);
        ptr += sizeof(size_t);
        
        uint8_t flags = *ptr++;
        binding.is_input = (flags & 0x01) != 0;
        binding.is_output = (flags & 0x02) != 0;
        binding.is_constant = (flags & 0x04) != 0;
        
        model.tensors.push_back(std::move(binding));
    }
    
    // Read constant data
    uint32_t constant_data_size = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += sizeof(uint32_t);
    
    if (constant_data_size > 0) {
        model.constant_data.assign(ptr, ptr + constant_data_size);
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::load_cmx_text(
    const char* text,
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    std::istringstream ss(text);
    std::string line;
    
    // Simple text format parser
    // Format: op_type node_name input1,input2 output1,output2
    uint32_t execution_order = 0;
    
    while (std::getline(ss, line)) {
        if (line.empty() || line[0] == '#') {
            continue; // Skip empty lines and comments
        }
        
        std::istringstream line_ss(line);
        std::string op_type, node_name, inputs_str, outputs_str;
        
        if (!(line_ss >> op_type >> node_name >> inputs_str >> outputs_str)) {
            continue; // Skip malformed lines
        }
        
        cmx_op_node node;
        node.op_type = op_type;
        node.node_name = node_name;
        node.execution_order = execution_order++;
        
        // Parse input IDs (simplified - assumes numeric IDs)
        std::istringstream inputs_ss(inputs_str);
        std::string input_id;
        while (std::getline(inputs_ss, input_id, ',')) {
            if (!input_id.empty()) {
                node.input_ids.push_back(std::stoul(input_id));
            }
        }
        
        // Parse output IDs
        std::istringstream outputs_ss(outputs_str);
        std::string output_id;
        while (std::getline(outputs_ss, output_id, ',')) {
            if (!output_id.empty()) {
                node.output_ids.push_back(std::stoul(output_id));
            }
        }
        
        model.nodes.push_back(std::move(node));
    }
    
    // Create basic tensor bindings (simplified)
    uint32_t max_tensor_id = 0;
    for (const auto& node : model.nodes) {
        for (uint32_t id : node.input_ids) {
            max_tensor_id = std::max(max_tensor_id, id);
        }
        for (uint32_t id : node.output_ids) {
            max_tensor_id = std::max(max_tensor_id, id);
        }
    }
    
    model.tensors.reserve(max_tensor_id + 1);
    for (uint32_t i = 0; i <= max_tensor_id; ++i) {
        cmx_tensor_binding binding;
        binding.tensor_id = i;
        binding.name = "tensor_" + std::to_string(i);
        binding.dtype = cmx_tensor_dtype::FLOAT32;
        binding.shape = {1}; // Default shape
        binding.is_input = false;
        binding.is_output = false;
        binding.is_constant = false;
        
        model.tensors.push_back(std::move(binding));
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::infer_shapes(
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    // Simplified shape inference - would need full implementation
    // based on operation types and input shapes
    
    for (auto& node : model.nodes) {
        // Shape inference would depend on specific operation types
        // This is a placeholder for the actual implementation
        if (node.op_type == "Conv2D") {
            // Implement Conv2D shape inference
        } else if (node.op_type == "Dense") {
            // Implement Dense layer shape inference
        }
        // Add more operation types as needed
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::optimize_layouts(
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    // Layout optimization implementation
    // Would optimize tensor layouts for better memory access patterns
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::fold_constants(
    cmx_loaded_model& model,
    const cmx_loader_config& config
) {
    // Constant folding implementation
    // Would evaluate constant operations at load time
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::validate_connections(const cmx_loaded_model& model) {
    // Validate that all tensor IDs referenced by nodes exist
    for (const auto& node : model.nodes) {
        for (uint32_t input_id : node.input_ids) {
            bool found = false;
            for (const auto& tensor : model.tensors) {
                if (tensor.tensor_id == input_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return cmx_status::INVALID_MODEL;
            }
        }
        
        for (uint32_t output_id : node.output_ids) {
            bool found = false;
            for (const auto& tensor : model.tensors) {
                if (tensor.tensor_id == output_id) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return cmx_status::INVALID_MODEL;
            }
        }
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::topological_sort(cmx_loaded_model& model) {
    // Simple topological sort based on execution_order
    std::sort(model.nodes.begin(), model.nodes.end(),
        [](const cmx_op_node& a, const cmx_op_node& b) {
            return a.execution_order < b.execution_order;
        });
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_op_loader::resolve_tensor_shapes(
    const cmx_op_node& node,
    const std::vector<cmx_tensor_binding>& tensors
) {
    // Tensor shape resolution implementation
    // Would resolve shapes based on operation semantics
    return cmx_status::SUCCESS;
}

// Format detector implementations
bool cmx_format_detector::is_cmx_binary(const uint8_t* buffer, size_t size) {
    return size >= sizeof(CMX_BINARY_MAGIC) &&
           memcmp(buffer, CMX_BINARY_MAGIC, sizeof(CMX_BINARY_MAGIC)) == 0;
}

bool cmx_format_detector::is_cmx_text(const char* text) {
    // Simple heuristic - look for operation keywords
    return strstr(text, "Conv2D") != nullptr ||
           strstr(text, "Dense") != nullptr ||
           strstr(text, "ReLU") != nullptr;
}

bool cmx_format_detector::is_onnx(const uint8_t* buffer, size_t size) {
    return size >= sizeof(ONNX_MAGIC) &&
           memcmp(buffer, ONNX_MAGIC, sizeof(ONNX_MAGIC)) == 0;
}

bool cmx_format_detector::is_tflite(const uint8_t* buffer, size_t size) {
    return size >= sizeof(TFLITE_MAGIC) &&
           memcmp(buffer, TFLITE_MAGIC, sizeof(TFLITE_MAGIC)) == 0;
}

} // namespace cmx