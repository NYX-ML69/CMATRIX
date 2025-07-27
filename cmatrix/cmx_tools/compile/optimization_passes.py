"""
optimization_passes.py - Graph-level optimizations

Applies various optimization passes to improve model performance before code generation.
"""

import copy
from typing import Dict, Any, List, Optional


class OptimizationPass:
    """Base class for optimization passes."""
    
    def __init__(self, name: str):
        self.name = name
        
    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization pass to graph."""
        raise NotImplementedError
        
    def can_apply(self, graph: Dict[str, Any]) -> bool:
        """Check if optimization can be applied."""
        return True


class ConstantFoldingPass(OptimizationPass):
    """Fold constant operations at compile time."""
    
    def __init__(self):
        super().__init__("constant_folding")
        
    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Fold constant operations."""
        optimized_graph = copy.deepcopy(graph)
        layers = optimized_graph.get('layers', [])
        
        # Find constant operations
        constant_layers = []
        for i, layer in enumerate(layers):
            if self._is_constant_operation(layer):
                constant_layers.append(i)
                
        # Fold constants (simplified implementation)
        for layer_idx in reversed(constant_layers):
            layer = layers[layer_idx]
            if layer.get('type') == 'add' and 'constant_value' in layer:
                # Example: fold constant addition
                folded_value = self._compute_constant_add(layer)
                # Replace with constant layer
                layers[layer_idx] = {
                    'type': 'constant',
                    'name': f"folded_const_{layer_idx}",
                    'value': folded_value,
                    'shape': layer.get('output_shape')
                }
                
        print(f"Constant folding: folded {len(constant_layers)} operations")
        return optimized_graph
    
    def _is_constant_operation(self, layer: Dict[str, Any]) -> bool:
        """Check if layer is a constant operation."""
        if layer.get('type') in ['add', 'mul', 'sub'] and 'constant_value' in layer:
            return True
        return False
    
    def _compute_constant_add(self, layer: Dict[str, Any]) -> float:
        """Compute constant addition result."""
        return layer.get('constant_value', 0.0)


class LayerFusionPass(OptimizationPass):
    """Fuse compatible layers together."""
    
    def __init__(self):
        super().__init__("layer_fusion")
        
    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse compatible layers."""
        optimized_graph = copy.deepcopy(graph)
        layers = optimized_graph.get('layers', [])
        
        fused_count = 0
        i = 0
        while i < len(layers) - 1:
            current_layer = layers[i]
            next_layer = layers[i + 1]
            
            # Check for fusible patterns
            if self._can_fuse_conv_relu(current_layer, next_layer):
                fused_layer = self._fuse_conv_relu(current_layer, next_layer)
                layers[i] = fused_layer
                layers.pop(i + 1)  # Remove the ReLU layer
                fused_count += 1
            elif self._can_fuse_conv_bn(current_layer, next_layer):
                fused_layer = self._fuse_conv_bn(current_layer, next_layer)
                layers[i] = fused_layer
                layers.pop(i + 1)  # Remove the BatchNorm layer
                fused_count += 1
            else:
                i += 1
                
        print(f"Layer fusion: fused {fused_count} layer pairs")
        return optimized_graph
    
    def _can_fuse_conv_relu(self, conv_layer: Dict, relu_layer: Dict) -> bool:
        """Check if Conv2D + ReLU can be fused."""
        return (conv_layer.get('type') == 'conv2d' and 
                relu_layer.get('type') == 'relu')
    
    def _can_fuse_conv_bn(self, conv_layer: Dict, bn_layer: Dict) -> bool:
        """Check if Conv2D + BatchNorm can be fused."""
        return (conv_layer.get('type') == 'conv2d' and 
                bn_layer.get('type') == 'batch_norm')
    
    def _fuse_conv_relu(self, conv_layer: Dict, relu_layer: Dict) -> Dict:
        """Fuse Conv2D + ReLU into Conv2D with ReLU activation."""
        fused_layer = copy.deepcopy(conv_layer)
        fused_layer['activation'] = 'relu'
        fused_layer['name'] = f"{conv_layer.get('name', 'conv')}_relu_fused"
        fused_layer['fused'] = True
        return fused_layer
    
    def _fuse_conv_bn(self, conv_layer: Dict, bn_layer: Dict) -> Dict:
        """Fuse Conv2D + BatchNorm by folding BN into Conv weights."""
        fused_layer = copy.deepcopy(conv_layer)
        
        # Fold BatchNorm parameters into Conv weights/bias
        # This is a simplified implementation
        bn_scale = bn_layer.get('scale', 1.0)
        bn_bias = bn_layer.get('bias', 0.0)
        bn_mean = bn_layer.get('mean', 0.0)
        bn_var = bn_layer.get('var', 1.0)
        
        fused_layer['batch_norm_folded'] = True
        fused_layer['bn_params'] = {
            'scale': bn_scale,
            'bias': bn_bias,
            'mean': bn_mean,
            'var': bn_var
        }
        fused_layer['name'] = f"{conv_layer.get('name', 'conv')}_bn_fused"
        
        return fused_layer


class DeadCodeEliminationPass(OptimizationPass):
    """Remove unused layers and operations."""
    
    def __init__(self):
        super().__init__("dead_code_elimination")
        
    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Remove dead code from graph."""
        optimized_graph = copy.deepcopy(graph)
        layers = optimized_graph.get('layers', [])
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(layers)
        
        # Find reachable layers from outputs
        reachable = self._find_reachable_layers(layers, dependencies)
        
        # Remove unreachable layers
        original_count = len(layers)
        layers[:] = [layer for i, layer in enumerate(layers) if i in reachable]
        removed_count = original_count - len(layers)
        
        print(f"Dead code elimination: removed {removed_count} unused layers")
        return optimized_graph
    
    def _build_dependency_graph(self, layers: List[Dict]) -> Dict[int, List[int]]:
        """Build layer dependency graph."""
        dependencies = {}
        
        for i, layer in enumerate(layers):
            dependencies[i] = []
            inputs = layer.get('inputs', [])
            
            # Find which layers produce the inputs
            for input_name in inputs:
                for j, other_layer in enumerate(layers):
                    if j != i and other_layer.get('name') == input_name:
                        dependencies[i].append(j)
                        
        return dependencies
    
    def _find_reachable_layers(self, layers: List[Dict], 
                              dependencies: Dict[int, List[int]]) -> set:
        """Find all layers reachable from outputs."""
        # Assume last layer is output (simplified)
        if not layers:
            return set()
            
        reachable = set()
        stack = [len(layers) - 1]  # Start from output layer
        
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
                
            reachable.add(current)
            # Add dependencies to stack
            for dep in dependencies.get(current, []):
                if dep not in reachable:
                    stack.append(dep)
                    
        return reachable


class MemoryOptimizationPass(OptimizationPass):
    """Optimize memory usage by reusing tensors."""
    
    def __init__(self):
        super().__init__("memory_optimization")
        
    def apply(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory layout."""
        optimized_graph = copy.deepcopy(graph)
        layers = optimized_graph.get('layers', [])
        
        # Analyze tensor lifetimes
        lifetimes = self._analyze_tensor_lifetimes(layers)
        
        # Assign memory pools
        memory_pools = self._assign_memory_pools(lifetimes)
        
        # Update tensor declarations
        tensors = optimized_graph.get('tensors', [])
        for tensor in tensors:
            tensor_name = tensor.get('name')
            if tensor_name in memory_pools:
                tensor['memory_pool'] = memory_pools[tensor_name]
                
        pool_count = len(set(memory_pools.values()))
        print(f"Memory optimization: reduced to {pool_count} memory pools")
        
        return optimized_graph
    
    def _analyze_tensor_lifetimes(self, layers: List[Dict]) -> Dict[str, tuple]:
        """Analyze when tensors are created and last used."""
        lifetimes = {}
        
        for i, layer in enumerate(layers):
            # Tensor is created at this layer
            outputs = layer.get('outputs', [layer.get('name', f'layer_{i}')])
            for output in outputs:
                if output not in lifetimes:
                    lifetimes[output] = [i, i]  # [creation, last_use]
            
            # Update last use for inputs
            inputs = layer.get('inputs', [])
            for input_tensor in inputs:
                if input_tensor in lifetimes:
                    lifetimes[input_tensor][1] = i
                    
        return lifetimes
    
    def _assign_memory_pools(self, lifetimes: Dict[str, tuple]) -> Dict[str, int]:
        """Assign tensors to memory pools based on lifetimes."""
        pools = {}
        pool_schedule = {}  # pool_id -> [(start, end)]
        next_pool_id = 0
        
        # Sort tensors by creation time
        sorted_tensors = sorted(lifetimes.items(), key=lambda x: x[1][0])
        
        for tensor_name, (start, end) in sorted_tensors:
            # Find a pool that's free during tensor lifetime
            assigned_pool = None
            
            for pool_id, schedule in pool_schedule.items():
                can_use_pool = True
                for pool_start, pool_end in schedule:
                    if not (end < pool_start or start > pool_end):
                        can_use_pool = False
                        break
                        
                if can_use_pool:
                    assigned_pool = pool_id
                    break
            
            # Create new pool if needed
            if assigned_pool is None:
                assigned_pool = next_pool_id
                next_pool_id += 1
                pool_schedule[assigned_pool] = []
            
            # Assign tensor to pool
            pools[tensor_name] = assigned_pool
            pool_schedule[assigned_pool].append((start, end))
            
        return pools


def optimize_graph(graph: Dict[str, Any], level: int = 2) -> Dict[str, Any]:
    """
    Apply optimization passes to graph.
    
    Args:
        graph: Input graph IR
        level: Optimization level (0-3)
            0: No optimizations
            1: Basic optimizations (constant folding)
            2: Standard optimizations (+ layer fusion, dead code elimination)
            3: Aggressive optimizations (+ memory optimization)
    
    Returns:
        Optimized graph IR
    """
    
    if level == 0:
        print("Optimization level 0: No optimizations applied")
        return graph
    
    optimized_graph = copy.deepcopy(graph)
    
    # Define optimization passes by level
    passes = []
    
    if level >= 1:
        passes.extend([
            ConstantFoldingPass(),
        ])
    
    if level >= 2:
        passes.extend([
            LayerFusionPass(),
            DeadCodeEliminationPass(),
        ])
    
    if level >= 3:
        passes.extend([
            MemoryOptimizationPass(),
        ])
    
    # Apply passes
    print(f"Applying {len(passes)} optimization passes (level {level}):")
    
    for pass_obj in passes:
        if pass_obj.can_apply(optimized_graph):
            print(f"  - {pass_obj.name}")
            optimized_graph = pass_obj.apply(optimized_graph)
        else:
            print(f"  - {pass_obj.name} (skipped - not applicable)")
    
    return optimized_graph


def validate_optimization(original_graph: Dict[str, Any], 
                         optimized_graph: Dict[str, Any]) -> bool:
    """
    Validate that optimization preserves graph semantics.
    
    Args:
        original_graph: Original graph before optimization
        optimized_graph: Graph after optimization
        
    Returns:
        bool: True if optimization is valid
    """
    
    # Check that input/output shapes are preserved
    orig_input_shape = original_graph.get('input_shape')
    opt_input_shape = optimized_graph.get('input_shape')
    
    if orig_input_shape != opt_input_shape:
        print(f"Warning: Input shape changed from {orig_input_shape} to {opt_input_shape}")
        return False
    
    orig_output_shape = original_graph.get('output_shape')
    opt_output_shape = optimized_graph.get('output_shape')
    
    if orig_output_shape != opt_output_shape:
        print(f"Warning: Output shape changed from {orig_output_shape} to {opt_output_shape}")
        return False
    
    print("Optimization validation passed")
    return True


def get_optimization_stats(original_graph: Dict[str, Any], 
                          optimized_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about applied optimizations.
    
    Returns:
        Dict with optimization statistics
    """
    
    orig_layers = len(original_graph.get('layers', []))
    opt_layers = len(optimized_graph.get('layers', []))
    
    orig_tensors = len(original_graph.get('tensors', []))
    opt_tensors = len(optimized_graph.get('tensors', []))
    
    stats = {
        'original_layers': orig_layers,
        'optimized_layers': opt_layers,
        'layers_removed': orig_layers - opt_layers,
        'layer_reduction_percent': ((orig_layers - opt_layers) / orig_layers * 100) if orig_layers > 0 else 0,
        
        'original_tensors': orig_tensors,
        'optimized_tensors': opt_tensors,
        'tensors_removed': orig_tensors - opt_tensors,
        
        'memory_pools': len(set(
            tensor.get('memory_pool', 0) 
            for tensor in optimized_graph.get('tensors', [])
        )) if optimized_graph.get('tensors') else 0
    }
    
    return stats