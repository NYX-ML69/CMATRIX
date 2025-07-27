#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>

namespace cmx {
namespace graph {

// Forward declarations
class CMXGraph;
class CMXNode;

/// @brief Tensor shape representation
struct TensorShape {
    static constexpr size_t MAX_DIMS = 8;
    
    uint32_t dims[MAX_DIMS];
    uint8_t rank;
    
    TensorShape() : rank(0) {
        for (size_t i = 0; i < MAX_DIMS; ++i) {
            dims[i] = 0;
        }
    }
    
    TensorShape(std::initializer_list<uint32_t> shape) : rank(0) {
        for (auto dim : shape) {
            if (rank < MAX_DIMS) {
                dims[rank++] = dim;
            }
        }
    }
    
    uint32_t& operator[](size_t index) { return dims[index]; }
    const uint32_t& operator[](size_t index) const { return dims[index]; }
    
    size_t total_elements() const {
        size_t total = 1;
        for (size_t i = 0; i < rank; ++i) {
            total *= dims[i];
        }
        return total;
    }
    
    bool is_valid() const { return rank > 0; }
    bool is_scalar() const { return rank == 1 && dims[0] == 1; }
};

/// @brief Shape inference context for operations
struct ShapeInferenceContext {
    std::vector<TensorShape> input_shapes;
    std::vector<TensorShape> output_shapes;
    std::unordered_map<std::string, int32_t> attributes_int;
    std::unordered_map<std::string, float> attributes_float;
    
    void clear() {
        input_shapes.clear();
        output_shapes.clear();
        attributes_int.clear();
        attributes_float.clear();
    }
};

/// @brief Shape inference function signature
using ShapeInferenceFunc = std::function<bool(ShapeInferenceContext&)>;

/// @brief Shape inference result
enum class ShapeInferenceResult {
    SUCCESS,
    FAILED_INVALID_INPUT,
    FAILED_UNSUPPORTED_OP,
    FAILED_INCOMPATIBLE_SHAPES,
    FAILED_MISSING_ATTRIBUTES
};

/// @brief Main shape inference engine
class CMXShapeInference {
public:
    CMXShapeInference();
    ~CMXShapeInference() = default;
    
    /// @brief Register shape inference function for an operation type
    /// @param op_type Operation type identifier
    /// @param func Shape inference function
    void register_shape_function(const std::string& op_type, ShapeInferenceFunc func);
    
    /// @brief Infer shapes for entire graph
    /// @param graph Input graph to infer shapes for
    /// @return Result of shape inference
    ShapeInferenceResult infer_shapes(CMXGraph& graph);
    
    /// @brief Infer shape for a single node
    /// @param node Node to infer shapes for
    /// @param context Shape inference context
    /// @return Result of shape inference
    ShapeInferenceResult infer_node_shape(const CMXNode& node, ShapeInferenceContext& context);
    
    /// @brief Validate tensor shapes compatibility
    /// @param shape1 First tensor shape
    /// @param shape2 Second tensor shape
    /// @return True if shapes are compatible
    static bool are_shapes_compatible(const TensorShape& shape1, const TensorShape& shape2);
    
    /// @brief Get broadcast result shape
    /// @param shape1 First input shape
    /// @param shape2 Second input shape
    /// @param result Output broadcast shape
    /// @return True if broadcast is valid
    static bool get_broadcast_shape(const TensorShape& shape1, const TensorShape& shape2, TensorShape& result);
    
private:
    std::unordered_map<std::string, ShapeInferenceFunc> shape_functions_;
    
    /// @brief Register built-in shape functions
    void register_builtin_shapes();
    
    /// @brief Built-in shape functions
    static bool infer_conv2d_shape(ShapeInferenceContext& ctx);
    static bool infer_pool_shape(ShapeInferenceContext& ctx);
    static bool infer_fully_connected_shape(ShapeInferenceContext& ctx);
    static bool infer_elementwise_shape(ShapeInferenceContext& ctx);
    static bool infer_reshape_shape(ShapeInferenceContext& ctx);
    static bool infer_concat_shape(ShapeInferenceContext& ctx);
    static bool infer_transpose_shape(ShapeInferenceContext& ctx);
};

} // namespace graph
} // namespace cmx