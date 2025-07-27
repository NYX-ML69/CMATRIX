"""
code_generator.py - C/C++ code generation from optimized graph IR

Converts optimized/quantized graphs into embedded-compatible C++ source code.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from string import Template

from .backend_flags import get_backend_flags, get_template_name


class CodeGenerator:
    """Generates C++ code from graph IR using templates."""
    
    def __init__(self, target: str):
        self.target = target
        self.backend_config = get_backend_flags(target)
        self.template_dir = Path(__file__).parent / "templates"
        
    def _load_template(self) -> str:
        """Load the appropriate template for the target."""
        template_name = get_template_name(self.target)
        template_path = self.template_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
            
        with open(template_path, 'r') as f:
            return f.read()
    
    def _generate_includes(self) -> str:
        """Generate #include directives."""
        includes = self.backend_config.get('includes', [])
        include_lines = []
        
        for inc in includes:
            if inc.endswith('.h'):
                include_lines.append(f'#include "{inc}"')
            else:
                include_lines.append(f'#include <{inc}>')
                
        return '\n'.join(include_lines)
    
    def _generate_defines(self) -> str:
        """Generate #define directives."""
        defines = self.backend_config.get('defines', [])
        define_lines = []
        
        for define in defines:
            define_lines.append(f'#define {define}')
            
        # Add memory constraints
        memory = self.backend_config
        define_lines.extend([
            f'#define STACK_SIZE {memory.get("stack_size", 4096)}',
            f'#define HEAP_SIZE {memory.get("heap_size", 8192)}',
            f'#define ALIGNMENT {memory.get("alignment", 4)}'
        ])
        
        return '\n'.join(define_lines)
    
    def _generate_tensor_declarations(self, graph_ir: Dict[str, Any]) -> str:
        """Generate static tensor declarations."""
        declarations = []
        
        # Extract tensor info from graph (placeholder implementation)
        tensors = graph_ir.get('tensors', [])
        
        for tensor in tensors:
            name = tensor.get('name', 'unknown')
            shape = tensor.get('shape', [1])
            dtype = tensor.get('dtype', 'float')
            
            # Calculate size
            size = 1
            for dim in shape:
                size *= dim
                
            # Map dtype to C++ type
            cpp_type = {
                'float': 'float',
                'int8': 'int8_t',
                'uint8': 'uint8_t',
                'int16': 'int16_t',
                'int32': 'int32_t'
            }.get(dtype, 'float')
            
            declarations.append(f'alignas({self.backend_config["alignment"]}) static {cpp_type} {name}[{size}];')
        
        return '\n'.join(declarations)
    
    def _generate_layer_functions(self, graph_ir: Dict[str, Any]) -> str:
        """Generate layer execution functions."""
        functions = []
        layers = graph_ir.get('layers', [])
        
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'unknown')
            func_name = f'layer_{i}_{layer_type}'
            
            if layer_type == 'conv2d':
                functions.append(self._generate_conv2d_function(func_name, layer))
            elif layer_type == 'relu':
                functions.append(self._generate_relu_function(func_name, layer))
            elif layer_type == 'dense':
                functions.append(self._generate_dense_function(func_name, layer))
            else:
                functions.append(self._generate_generic_function(func_name, layer))
        
        return '\n\n'.join(functions)
    
    def _generate_conv2d_function(self, name: str, layer: Dict) -> str:
        """Generate Conv2D layer function."""
        return f"""
void {name}(const float* input, float* output, const float* weights, const float* bias) {{
    // Conv2D implementation for {layer.get('name', 'conv')}
    // Kernel size: {layer.get('kernel_size', [3, 3])}
    // Stride: {layer.get('stride', [1, 1])}
    // TODO: Implement optimized convolution
    cmx_conv2d(input, output, weights, bias, 
               {layer.get('input_channels', 1)}, 
               {layer.get('output_channels', 1)},
               {layer.get('kernel_size', [3, 3])[0]});
}}"""
    
    def _generate_relu_function(self, name: str, layer: Dict) -> str:
        """Generate ReLU activation function."""
        return f"""
void {name}(float* data, int size) {{
    // ReLU activation for {layer.get('name', 'relu')}
    cmx_relu(data, size);
}}"""
    
    def _generate_dense_function(self, name: str, layer: Dict) -> str:
        """Generate Dense/Linear layer function."""
        return f"""
void {name}(const float* input, float* output, const float* weights, const float* bias) {{
    // Dense layer for {layer.get('name', 'dense')}
    // Input size: {layer.get('input_size', 1)}
    // Output size: {layer.get('output_size', 1)}
    cmx_dense(input, output, weights, bias, 
              {layer.get('input_size', 1)}, 
              {layer.get('output_size', 1)});
}}"""
    
    def _generate_generic_function(self, name: str, layer: Dict) -> str:
        """Generate generic layer function."""
        return f"""
void {name}(const float* input, float* output) {{
    // Generic layer: {layer.get('type', 'unknown')}
    // TODO: Implement {layer.get('name', 'layer')}
}}"""
    
    def _generate_inference_loop(self, graph_ir: Dict[str, Any]) -> str:
        """Generate main inference execution loop."""
        layers = graph_ir.get('layers', [])
        loop_body = []
        
        loop_body.append("    // Initialize input tensor")
        loop_body.append("    // Copy input data to input_tensor")
        
        for i, layer in enumerate(layers):
            layer_type = layer.get('type', 'unknown')
            func_name = f'layer_{i}_{layer_type}'
            
            if layer_type == 'conv2d':
                loop_body.append(f"    {func_name}(input_tensor, temp_tensor, weights_{i}, bias_{i});")
            elif layer_type == 'relu':
                loop_body.append(f"    {func_name}(temp_tensor, tensor_size);")
            else:
                loop_body.append(f"    {func_name}(input_tensor, output_tensor);")
        
        loop_body.append("    // Output ready in output_tensor")
        
        return '\n'.join(loop_body)
    
    def generate(self, graph_ir: Dict[str, Any]) -> str:
        """Generate complete C++ code from graph IR."""
        template_content = self._load_template()
        
        # Generate code sections
        includes = self._generate_includes()
        defines = self._generate_defines()
        tensor_decls = self._generate_tensor_declarations(graph_ir)
        layer_functions = self._generate_layer_functions(graph_ir)
        inference_loop = self._generate_inference_loop(graph_ir)
        
        # Substitute into template
        template = Template(template_content)
        generated_code = template.safe_substitute(
            includes=includes,
            defines=defines,
            tensor_declarations=tensor_decls,
            layer_functions=layer_functions,
            inference_loop=inference_loop,
            target=self.target.upper(),
            model_name=graph_ir.get('name', 'model')
        )
        
        return generated_code


def generate_code(graph_ir: Dict[str, Any], target: str, output_dir: str = './output') -> str:
    """
    Generate C++ code from optimized graph IR.
    
    Args:
        graph_ir: Optimized and quantized graph representation
        target: Target backend name
        output_dir: Directory to save generated code
        
    Returns:
        str: Path to generated C++ file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate code
    generator = CodeGenerator(target)
    cpp_code = generator.generate(graph_ir)
    
    # Write to file
    model_name = graph_ir.get('name', 'model')
    output_filename = f"{model_name}_{target}.cpp"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        f.write(cpp_code)
    
    # Also generate header file
    header_path = output_path.replace('.cpp', '.h')
    header_content = generate_header(graph_ir, target)
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    return output_path


def generate_header(graph_ir: Dict[str, Any], target: str) -> str:
    """Generate accompanying header file."""
    model_name = graph_ir.get('name', 'model')
    guard_name = f"{model_name.upper()}_{target.upper()}_H"
    
    header = f"""#ifndef {guard_name}
#define {guard_name}

#ifdef __cplusplus
extern "C" {{
#endif

// Model inference function
int {model_name}_inference(const float* input, float* output);

// Initialization and cleanup
int {model_name}_init(void);
void {model_name}_cleanup(void);

// Model metadata
const char* {model_name}_get_version(void);
int {model_name}_get_input_size(void);
int {model_name}_get_output_size(void);

#ifdef __cplusplus
}}
#endif

#endif // {guard_name}
"""
    
    return header


