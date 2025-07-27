#include "cmx_shape_inference.hpp"
#include "cmx_graph.hpp"
#include "cmx_node.hpp"
#include <algorithm>
#include <cmath>

namespace cmx {
namespace graph {

CMXShapeInference::CMXShapeInference() {
    register_builtin_shapes();
}

void CMXShapeInference::register_shape_function(const std::string& op_type, ShapeInferenceFunc func) {
    shape_functions_[op_type] = std::move(func);
}

ShapeInferenceResult CMXShapeInference::infer_shapes(CMXGraph& graph) {
    // TODO: Implement topological traversal of graph
    // For now, placeholder implementation
    return ShapeInferenceResult::SUCCESS;
}

ShapeInferenceResult CMXShapeInference::infer_node_shape(const CMXNode& node, ShapeInferenceContext& context) {
    // TODO: Get op_type from node
    std::string op_type = ""; // node.get_op_type();
    
    auto it = shape_functions_.find(op_type);
    if (it == shape_functions_.end()) {
        return ShapeInferenceResult::FAILED_UNSUPPORTED_OP;
    }
    
    if (!it->second(context)) {
        return ShapeInferenceResult::FAILED_INVALID_INPUT;
    }
    
    return ShapeInferenceResult::SUCCESS;
}

bool CMXShapeInference::are_shapes_compatible(const TensorShape& shape1, const TensorShape& shape2) {
    if (shape1.rank != shape2.rank) {
        return false;
    }
    
    for (size_t i = 0; i < shape1.rank; ++i) {
        if (shape1.dims[i] != shape2.dims[i]) {
            return false;
        }
    }
    
    return true;
}

bool CMXShapeInference::get_broadcast_shape(const TensorShape& shape1, const TensorShape& shape2, TensorShape& result) {
    result.rank = std::max(shape1.rank, shape2.rank);
    
    for (size_t i = 0; i < result.rank; ++i) {
        size_t idx1 = (i < shape1.rank) ? (shape1.rank - 1 - i) : 0;
        size_t idx2 = (i < shape2.rank) ? (shape2.rank - 1 - i) : 0;
        size_t res_idx = result.rank - 1 - i;
        
        uint32_t dim1 = (i < shape1.rank) ? shape1.dims[idx1] : 1;
        uint32_t dim2 = (i < shape2.rank) ? shape2.dims[idx2] : 1;
        
        if (dim1 == dim2) {
            result.dims[res_idx] = dim1;
        } else if (dim1 == 1) {
            result.dims[res_idx] = dim2;
        } else if (dim2 == 1) {
            result.dims[res_idx] = dim1;
        } else {
            return false; // Incompatible dimensions
        }
    }
    
    return true;
}

void CMXShapeInference::register_builtin_shapes() {
    register_shape_function("Conv2D", infer_conv2d_shape);
    register_shape_function("MaxPool2D", infer_pool_shape);
    register_shape_function("AvgPool2D", infer_pool_shape);
    register_shape_function("FullyConnected", infer_fully_connected_shape);
    register_shape_function("Add", infer_elementwise_shape);
    register_shape_function("Mul", infer_elementwise_shape);
    register_shape_function("Reshape", infer_reshape_shape);
    register_shape_function("Concat", infer_concat_shape);
    register_shape_function("Transpose", infer_transpose_shape);
}

bool CMXShapeInference::infer_conv2d_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return false;
    }
    
    const TensorShape& input = ctx.input_shapes[0];
    if (input.rank != 4) {
        return false;
    }
    
    // Get convolution parameters
    int32_t kernel_h = ctx.attributes_int.count("kernel_h") ? ctx.attributes_int["kernel_h"] : 3;
    int32_t kernel_w = ctx.attributes_int.count("kernel_w") ? ctx.attributes_int["kernel_w"] : 3;
    int32_t stride_h = ctx.attributes_int.count("stride_h") ? ctx.attributes_int["stride_h"] : 1;
    int32_t stride_w = ctx.attributes_int.count("stride_w") ? ctx.attributes_int["stride_w"] : 1;
    int32_t pad_h = ctx.attributes_int.count("pad_h") ? ctx.attributes_int["pad_h"] : 0;
    int32_t pad_w = ctx.attributes_int.count("pad_w") ? ctx.attributes_int["pad_w"] : 0;
    int32_t out_channels = ctx.attributes_int.count("out_channels") ? ctx.attributes_int["out_channels"] : 1;
    
    // Assume NHWC format
    uint32_t batch = input.dims[0];
    uint32_t in_h = input.dims[1];
    uint32_t in_w = input.dims[2];
    
    uint32_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    uint32_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    
    TensorShape output_shape({batch, out_h, out_w, static_cast<uint32_t>(out_channels)});
    ctx.output_shapes.push_back(output_shape);
    
    return true;
}

bool CMXShapeInference::infer_pool_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return false;
    }
    
    const TensorShape& input = ctx.input_shapes[0];
    if (input.rank != 4) {
        return false;
    }
    
    int32_t pool_h = ctx.attributes_int.count("pool_h") ? ctx.attributes_int["pool_h"] : 2;
    int32_t pool_w = ctx.attributes_int.count("pool_w") ? ctx.attributes_int["pool_w"] : 2;
    int32_t stride_h = ctx.attributes_int.count("stride_h") ? ctx.attributes_int["stride_h"] : 2;
    int32_t stride_w = ctx.attributes_int.count("stride_w") ? ctx.attributes_int["stride_w"] : 2;
    
    uint32_t batch = input.dims[0];
    uint32_t in_h = input.dims[1];
    uint32_t in_w = input.dims[2];
    uint32_t channels = input.dims[3];
    
    uint32_t out_h = (in_h - pool_h) / stride_h + 1;
    uint32_t out_w = (in_w - pool_w) / stride_w + 1;
    
    TensorShape output_shape({batch, out_h, out_w, channels});
    ctx.output_shapes.push_back(output_shape);
    
    return true;
}

bool CMXShapeInference::infer_fully_connected_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return false;
    }
    
    const TensorShape& input = ctx.input_shapes[0];
    int32_t output_size = ctx.attributes_int.count("output_size") ? ctx.attributes_int["output_size"] : 1;
    
    TensorShape output_shape({input.dims[0], static_cast<uint32_t>(output_size)});
    ctx.output_shapes.push_back(output_shape);
    
    return true;
}

bool CMXShapeInference::infer_elementwise_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return false;
    }
    
    TensorShape result_shape;
    if (!get_broadcast_shape(ctx.input_shapes[0], ctx.input_shapes[1], result_shape)) {
        return false;
    }
    
    ctx.output_shapes.push_back(result_shape);
    return true;
}

bool CMXShapeInference::infer_reshape_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return false;
    }
    
    // TODO: Parse target shape from attributes
    // For now, return input shape as placeholder
    ctx.output_shapes.push_back(ctx.input_shapes[0]);
    return true;
}

bool CMXShapeInference::infer_concat_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.size() < 2) {
        return false;
    }
    
    int32_t axis = ctx.attributes_int.count("axis") ? ctx.attributes_int["axis"] : 0;
    const TensorShape& first_shape = ctx.input_shapes[0];
    
    TensorShape result_shape = first_shape;
    
    // Sum dimensions along concatenation axis
    for (size_t i = 1; i < ctx.input_shapes.size(); ++i) {
        const TensorShape& shape = ctx.input_shapes[i];
        if (shape.rank != first_shape.rank) {
            return false;
        }
        
        // Check all dimensions except concat axis are same
        for (size_t j = 0; j < shape.rank; ++j) {
            if (j != static_cast<size_t>(axis) && shape.dims[j] != first_shape.dims[j]) {
                return false;
            }
        }
        
        result_shape.dims[axis] += shape.dims[axis];
    }
    
    ctx.output_shapes.push_back(result_shape);
    return true;
}

bool CMXShapeInference::infer_transpose_shape(ShapeInferenceContext& ctx) {
    if (ctx.input_shapes.empty()) {
        return false;
    }
    
    const TensorShape& input = ctx.input_shapes[0];
    TensorShape output_shape = input;
    
    // TODO: Parse permutation from attributes
    // For now, just reverse dimensions as placeholder
    for (size_t i = 0; i < input.rank; ++i) {
        output_shape.dims[i] = input.dims[input.rank - 1 - i];
    }
    
    ctx.output_shapes.push_back(output_shape);
    return true;
}

} // namespace graph
} // namespace cmx