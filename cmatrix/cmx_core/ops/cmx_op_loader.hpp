#pragma once

#include <string>
#include <cstdint>
#include <vector>

namespace cmx {

// Forward declarations
struct cmx_model;
struct cmx_tensor;
struct cmx_op_graph;
enum class cmx_status : uint8_t;
enum class cmx_tensor_dtype : uint8_t;

/**
 * @brief Model format types supported by loader
 */
enum class cmx_model_format : uint8_t {
    CMX_BINARY = 0,     // Native CMatrix binary format
    CMX_TEXT = 1,       // Human-readable text format
    ONNX = 2,           // ONNX model format
    TFLITE = 3,         // TensorFlow Lite format
    CUSTOM = 255        // Custom/plugin format
};

/**
 * @brief Loader configuration options
 */
struct cmx_loader_config {
    bool enable_shape_inference = true;
    bool enable_layout_optimization = true;
    bool enable_constant_folding = false;
    bool strict_type_checking = true;
    size_t max_graph_nodes = 1024;
    size_t max_tensor_count = 512;
    const char* custom_op_library = nullptr;
};

/**
 * @brief Tensor binding information for runtime
 */
struct cmx_tensor_binding {
    std::string name;
    uint32_t tensor_id;
    cmx_tensor_dtype dtype;
    std::vector<uint32_t> shape;
    size_t byte_offset;     // Offset in model data
    bool is_input;
    bool is_output;
    bool is_constant;
};

/**
 * @brief Operation node in loaded graph
 */
struct cmx_op_node {
    std::string op_type;
    std::string node_name;
    std::vector<uint32_t> input_ids;
    std::vector<uint32_t> output_ids;
    std::vector<uint8_t> op_params;  // Serialized operation parameters
    uint32_t execution_order;
};

/**
 * @brief Loaded model representation
 */
struct cmx_loaded_model {
    std::vector<cmx_op_node> nodes;
    std::vector<cmx_tensor_binding> tensors;
    std::vector<uint8_t> constant_data;
    std::string model_name;
    uint32_t model_version;
    cmx_model_format format;
};

/**
 * @brief Model loader class
 */
class cmx_op_loader {
public:
    /**
     * @brief Load model from file path
     */
    static cmx_status load_from_file(
        const char* file_path,
        cmx_loaded_model& model,
        const cmx_loader_config& config = {}
    );
    
    /**
     * @brief Load model from memory buffer
     */
    static cmx_status load_from_buffer(
        const uint8_t* buffer,
        size_t buffer_size,
        cmx_model_format format,
        cmx_loaded_model& model,
        const cmx_loader_config& config = {}
    );
    
    /**
     * @brief Load model from string (text formats)
     */
    static cmx_status load_from_string(
        const char* model_string,
        cmx_model_format format,
        cmx_loaded_model& model,
        const cmx_loader_config& config = {}
    );
    
    /**
     * @brief Convert loaded model to executable graph
     */
    static cmx_status create_executable_graph(
        const cmx_loaded_model& model,
        cmx_op_graph& graph
    );
    
    /**
     * @brief Bind runtime tensors to loaded model
     */
    static cmx_status bind_tensors(
        const cmx_loaded_model& model,
        cmx_tensor* input_tensors,
        size_t num_inputs,
        cmx_tensor* output_tensors,
        size_t num_outputs
    );
    
    /**
     * @brief Validate loaded model integrity
     */
    static cmx_status validate_model(const cmx_loaded_model& model);
    
    /**
     * @brief Get model metadata
     */
    static cmx_status get_model_info(
        const cmx_loaded_model& model,
        size_t& num_inputs,
        size_t& num_outputs,
        size_t& num_operations
    );
    
    /**
     * @brief Free loaded model resources
     */
    static void free_model(cmx_loaded_model& model);

private:
    /**
     * @brief Detect model format from file extension or magic bytes
     */
    static cmx_model_format detect_format(
        const char* file_path,
        const uint8_t* buffer = nullptr,
        size_t buffer_size = 0
    );
    
    /**
     * @brief Load native CMX binary format
     */
    static cmx_status load_cmx_binary(
        const uint8_t* buffer,
        size_t buffer_size,
        cmx_loaded_model& model,
        const cmx_loader_config& config
    );
    
    /**
     * @brief Load CMX text format
     */
    static cmx_status load_cmx_text(
        const char* text,
        cmx_loaded_model& model,
        const cmx_loader_config& config
    );
    
    /**
     * @brief Perform shape inference on loaded graph
     */
    static cmx_status infer_shapes(
        cmx_loaded_model& model,
        const cmx_loader_config& config
    );
    
    /**
     * @brief Optimize tensor layouts
     */
    static cmx_status optimize_layouts(
        cmx_loaded_model& model,
        const cmx_loader_config& config
    );
    
    /**
     * @brief Fold constant operations
     */
    static cmx_status fold_constants(
        cmx_loaded_model& model,
        const cmx_loader_config& config
    );
    
    /**
     * @brief Validate tensor connections
     */
    static cmx_status validate_connections(const cmx_loaded_model& model);
    
    /**
     * @brief Sort operations in topological order
     */
    static cmx_status topological_sort(cmx_loaded_model& model);
    
    /**
     * @brief Resolve tensor shapes from operation inputs/outputs
     */
    static cmx_status resolve_tensor_shapes(
        const cmx_op_node& node,
        const std::vector<cmx_tensor_binding>& tensors
    );
};

/**
 * @brief Model format detection utilities
 */
class cmx_format_detector {
public:
    static bool is_cmx_binary(const uint8_t* buffer, size_t size);
    static bool is_cmx_text(const char* text);
    static bool is_onnx(const uint8_t* buffer, size_t size);
    static bool is_tflite(const uint8_t* buffer, size_t size);
};

/**
 * @brief Macro for registering custom model format loaders
 */
#define CMX_REGISTER_FORMAT_LOADER(format, loader_fn) \
    static const bool _cmx_format_registered_##format = []() { \
        return cmx_op_loader::register_format_loader(format, loader_fn); \
    }()

} // namespace cmx

