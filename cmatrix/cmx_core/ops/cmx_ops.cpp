#include "cmx_ops.hpp"
#include "cmx_op_context.hpp"
#include "cmx_op_registry.hpp"
#include <cstring>

namespace cmx {

// Core operation implementations
cmx_status cmx_conv2d_execute(cmx_op_context& ctx) {
    // Basic Conv2D implementation
    if (ctx.input_count != 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Get input tensor, weight tensor, and output tensor
    auto* input = ctx.inputs[0];
    auto* weights = ctx.inputs[1];
    auto* output = ctx.outputs[0];
    
    if (!input || !weights || !output) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Simple convolution implementation (placeholder)
    // In real implementation, this would call optimized kernels
    return cmx_status::SUCCESS;
}

cmx_status cmx_relu_execute(cmx_op_context& ctx) {
    if (ctx.input_count != 1 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    auto* input = ctx.inputs[0];
    auto* output = ctx.outputs[0];
    
    if (!input || !output) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // ReLU: max(0, x)
    const float* in_data = static_cast<const float*>(input->data);
    float* out_data = static_cast<float*>(output->data);
    
    for (size_t i = 0; i < input->size; ++i) {
        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_dense_execute(cmx_op_context& ctx) {
    if (ctx.input_count < 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Dense layer: Y = X * W + b
    return cmx_status::SUCCESS;
}

cmx_status cmx_add_execute(cmx_op_context& ctx) {
    if (ctx.input_count != 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    auto* input1 = ctx.inputs[0];
    auto* input2 = ctx.inputs[1];
    auto* output = ctx.outputs[0];
    
    if (!input1 || !input2 || !output) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    const float* in1_data = static_cast<const float*>(input1->data);
    const float* in2_data = static_cast<const float*>(input2->data);
    float* out_data = static_cast<float*>(output->data);
    
    size_t size = std::min(input1->size, std::min(input2->size, output->size));
    for (size_t i = 0; i < size; ++i) {
        out_data[i] = in1_data[i] + in2_data[i];
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_sub_execute(cmx_op_context& ctx) {
    if (ctx.input_count != 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    auto* input1 = ctx.inputs[0];
    auto* input2 = ctx.inputs[1];
    auto* output = ctx.outputs[0];
    
    const float* in1_data = static_cast<const float*>(input1->data);
    const float* in2_data = static_cast<const float*>(input2->data);
    float* out_data = static_cast<float*>(output->data);
    
    size_t size = std::min(input1->size, std::min(input2->size, output->size));
    for (size_t i = 0; i < size; ++i) {
        out_data[i] = in1_data[i] - in2_data[i];
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_mul_execute(cmx_op_context& ctx) {
    if (ctx.input_count != 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    auto* input1 = ctx.inputs[0];
    auto* input2 = ctx.inputs[1];
    auto* output = ctx.outputs[0];
    
    const float* in1_data = static_cast<const float*>(input1->data);
    const float* in2_data = static_cast<const float*>(input2->data);
    float* out_data = static_cast<float*>(output->data);
    
    size_t size = std::min(input1->size, std::min(input2->size, output->size));
    for (size_t i = 0; i < size; ++i) {
        out_data[i] = in1_data[i] * in2_data[i];
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_div_execute(cmx_op_context& ctx) {
    if (ctx.input_count != 2 || ctx.output_count != 1) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    auto* input1 = ctx.inputs[0];
    auto* input2 = ctx.inputs[1];
    auto* output = ctx.outputs[0];
    
    const float* in1_data = static_cast<const float*>(input1->data);
    const float* in2_data = static_cast<const float*>(input2->data);
    float* out_data = static_cast<float*>(output->data);
    
    size_t size = std::min(input1->size, std::min(input2->size, output->size));
    for (size_t i = 0; i < size; ++i) {
        out_data[i] = in2_data[i] != 0.0f ? in1_data[i] / in2_data[i] : 0.0f;
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_maxpool2d_execute(cmx_op_context& ctx) {
    // MaxPool2D implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_avgpool2d_execute(cmx_op_context& ctx) {
    // AvgPool2D implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_batchnorm_execute(cmx_op_context& ctx) {
    // BatchNorm implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_softmax_execute(cmx_op_context& ctx) {
    // Softmax implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_reshape_execute(cmx_op_context& ctx) {
    // Reshape implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_transpose_execute(cmx_op_context& ctx) {
    // Transpose implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_concat_execute(cmx_op_context& ctx) {
    // Concat implementation placeholder
    return cmx_status::SUCCESS;
}

cmx_status cmx_split_execute(cmx_op_context& ctx) {
    // Split implementation placeholder
    return cmx_status::SUCCESS;
}

// Utility functions
const char* cmx_status_to_string(cmx_status status) {
    switch (status) {
        case cmx_status::SUCCESS: return "SUCCESS";
        case cmx_status::ERROR_INVALID_ARGS: return "ERROR_INVALID_ARGS";
        case cmx_status::ERROR_OUT_OF_MEMORY: return "ERROR_OUT_OF_MEMORY";
        case cmx_status::ERROR_UNSUPPORTED_OP: return "ERROR_UNSUPPORTED_OP";
        case cmx_status::ERROR_EXECUTION_FAILED: return "ERROR_EXECUTION_FAILED";
        case cmx_status::ERROR_INVALID_CONTEXT: return "ERROR_INVALID_CONTEXT";
        case cmx_status::ERROR_TENSOR_MISMATCH: return "ERROR_TENSOR_MISMATCH";
        default: return "UNKNOWN_ERROR";
    }
}

cmx_op_type cmx_string_to_op_type(const std::string& name) {
    if (name == "Conv2D") return cmx_op_type::CONV2D;
    if (name == "ReLU") return cmx_op_type::RELU;
    if (name == "Dense") return cmx_op_type::DENSE;
    if (name == "Add") return cmx_op_type::ADD;
    if (name == "Sub") return cmx_op_type::SUB;
    if (name == "Mul") return cmx_op_type::MUL;
    if (name == "Div") return cmx_op_type::DIV;
    if (name == "MaxPool2D") return cmx_op_type::MAXPOOL2D;
    if (name == "AvgPool2D") return cmx_op_type::AVGPOOL2D;
    if (name == "BatchNorm") return cmx_op_type::BATCHNORM;
    if (name == "Softmax") return cmx_op_type::SOFTMAX;
    if (name == "Reshape") return cmx_op_type::RESHAPE;
    if (name == "Transpose") return cmx_op_type::TRANSPOSE;
    if (name == "Concat") return cmx_op_type::CONCAT;
    if (name == "Split") return cmx_op_type::SPLIT;
    return cmx_op_type::CUSTOM;
}

const char* cmx_op_type_to_string(cmx_op_type type) {
    switch (type) {
        case cmx_op_type::CONV2D: return "Conv2D";
        case cmx_op_type::RELU: return "ReLU";
        case cmx_op_type::DENSE: return "Dense";
        case cmx_op_type::ADD: return "Add";
        case cmx_op_type::SUB: return "Sub";
        case cmx_op_type::MUL: return "Mul";
        case cmx_op_type::DIV: return "Div";
        case cmx_op_type::MAXPOOL2D: return "MaxPool2D";
        case cmx_op_type::AVGPOOL2D: return "AvgPool2D";
        case cmx_op_type::BATCHNORM: return "BatchNorm";
        case cmx_op_type::SOFTMAX: return "Softmax";
        case cmx_op_type::RESHAPE: return "Reshape";
        case cmx_op_type::TRANSPOSE: return "Transpose";
        case cmx_op_type::CONCAT: return "Concat";
        case cmx_op_type::SPLIT: return "Split";
        case cmx_op_type::CUSTOM: return "Custom";
        default: return "Unknown";
    }
}

// Core operations initialization
void cmx_init_core_ops() {
    // Register all core operations
    cmx_op conv2d_op = {"Conv2D", cmx_conv2d_execute, 2, 1, 0, false, 1};
    cmx_op relu_op = {"ReLU", cmx_relu_execute, 1, 1, 0, true, 1};
    cmx_op dense_op = {"Dense", cmx_dense_execute, 2, 1, 0, false, 1};
    cmx_op add_op = {"Add", cmx_add_execute, 2, 1, 0, false, 1};
    cmx_op sub_op = {"Sub", cmx_sub_execute, 2, 1, 0, false, 1};
    cmx_op mul_op = {"Mul", cmx_mul_execute, 2, 1, 0, false, 1};
    cmx_op div_op = {"Div", cmx_div_execute, 2, 1, 0, false, 1};
    cmx_op maxpool_op = {"MaxPool2D", cmx_maxpool2d_execute, 1, 1, 0, false, 1};
    cmx_op avgpool_op = {"AvgPool2D", cmx_avgpool2d_execute, 1, 1, 0, false, 1};
    cmx_op batchnorm_op = {"BatchNorm", cmx_batchnorm_execute, 3, 1, 0, false, 1};
    cmx_op softmax_op = {"Softmax", cmx_softmax_execute, 1, 1, 0, false, 1};
    cmx_op reshape_op = {"Reshape", cmx_reshape_execute, 1, 1, 1, false, 1};
    cmx_op transpose_op = {"Transpose", cmx_transpose_execute, 1, 1, 1, false, 1};
    cmx_op concat_op = {"Concat", cmx_concat_execute, 2, 1, 1, false, 1};
    cmx_op split_op = {"Split", cmx_split_execute, 1, 2, 1, false, 1};
    
    cmx_register_op("Conv2D", conv2d_op);
    cmx_register_op("ReLU", relu_op);
    cmx_register_op("Dense", dense_op);
    cmx_register_op("Add", add_op);
    cmx_register_op("Sub", sub_op);
    cmx_register_op("Mul", mul_op);
    cmx_register_op("Div", div_op);
    cmx_register_op("MaxPool2D", maxpool_op);
    cmx_register_op("AvgPool2D", avgpool_op);
    cmx_register_op("BatchNorm", batchnorm_op);
    cmx_register_op("Softmax", softmax_op);
    cmx_register_op("Reshape", reshape_op);
    cmx_register_op("Transpose", transpose_op);
    cmx_register_op("Concat", concat_op);
    cmx_register_op("Split", split_op);
}

} // namespace cmx