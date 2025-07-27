#include "cmx_graph_executor.hpp"
#include "cmx_op_registry.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace cmx {

// Constructor
cmx_graph_executor::cmx_graph_executor()
    : config_(cmx_default_graph_config())
    , stats_{}
    , graph_(nullptr)
    , executor_(nullptr)
    , is_loaded_(false)
    , is_optimized_(false)
    , current_memory_usage_(0)
    , peak_memory_usage_(0) {
    
    executor_ = new cmx_op_executor(config_.executor_config);
}

cmx_graph_executor::cmx_graph_executor(const cmx_graph_config& config)
    : config_(config)
    , stats_{}
    , graph_(nullptr)
    , executor_(nullptr)
    , is_loaded_(false)
    , is_optimized_(false)
    , current_memory_usage_(0)
    , peak_memory_usage_(0) {
    
    executor_ = new cmx_op_executor(config_.executor_config);
}

// Destructor
cmx_graph_executor::~cmx_graph_executor() {
    if (graph_) {
        cmx_destroy_graph(graph_);
    }
    delete executor_;
}

// Model loading (placeholder - cmx_model structure would be defined elsewhere)
cmx_status cmx_graph_executor::load(const cmx_model& model) {
    // This would convert a model to internal graph representation
    // For now, return success as placeholder
    is_loaded_ = true;
    return cmx_status::SUCCESS;
}

cmx_status cmx_graph_executor::load_from_graph(const cmx_graph& graph) {
    if (graph_) {
        cmx_destroy_graph(graph_);
    }
    
    // Create a copy of the graph
    cmx_status status = cmx_create_graph(&graph_, graph.max_nodes, graph.max_tensors);
    if (status != cmx_status::SUCCESS) {
        return status;
    }
    
    // Copy graph data
    std::memcpy(graph_->nodes, graph.nodes, graph.node_count * sizeof(cmx_graph_node));
    std::memcpy(graph_->tensors, graph.tensors, graph.tensor_count * sizeof(cmx_tensor));
    
    graph_->node_count = graph.node_count;
    graph_->tensor_count = graph.tensor_count;
    graph_->input_count = graph.input_count;
    graph_->output_count = graph.output_count;
    
    status = initialize_graph();
    if (status == cmx_status::SUCCESS) {
        is_loaded_ = true;
        
        if (config_.enable_optimization) {
            optimize_execution_order();
            optimize_memory_layout();
        }
    }
    
    return status;
}

// Graph execution
cmx_status cmx_graph_executor::run() {
    if (!is_loaded_ || !graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Validate graph before execution
    cmx_status status = validate_graph();
    if (status != cmx_status::SUCCESS) {
        return status;
    }
    
    // Schedule execution if not already done
    if (!is_optimized_) {
        status = schedule_execution();
        if (status != cmx_status::SUCCESS) {
            return status;
        }
    }
    
    // Start profiling if enabled
    uint64_t start_time = 0;
    if (config_.enable_profiling) {
        // Would get actual timestamp
    }
    
    // Execute nodes in order
    for (uint32_t i = 0; i < graph_->node_count; ++i) {
        uint32_t node_idx = graph_->execution_order[i];
        status = execute_node(&graph_->nodes[node_idx]);
        
        if (status != cmx_status::SUCCESS) {
            stats_.failed_executions++;
            return status;
        }
        
        stats_.total_nodes_executed++;
    }
    
    // End profiling if enabled
    if (config_.enable_profiling) {
        uint64_t end_time = 0; // Would get actual timestamp
        uint64_t execution_time = end_time - start_time;
        
        stats_.total_graphs_executed++;
        stats_.total_execution_time += execution_time;
        stats_.avg_graph_execution_time = stats_.total_execution_time / stats_.total_graphs_executed;
        
        // Update peak memory usage
        if (current_memory_usage_ > peak_memory_usage_) {
            peak_memory_usage_ = current_memory_usage_;
            stats_.memory_peak_usage = static_cast<uint32_t>(peak_memory_usage_);
        }
    }
    
    return cmx_status::SUCCESS;
}

cmx_status cmx_graph_executor::run(cmx_tensor* inputs, uint32_t input_count,
                                  cmx_tensor* outputs, uint32_t output_count) {
    if (!inputs || !outputs || input_count != graph_->input_count || 
        output_count != graph_->output_count) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Set input tensors
    for (uint32_t i = 0; i < input_count; ++i) {
        cmx_status status = set_input(i, &inputs[i]);
        if (status != cmx_status::SUCCESS) {
            return status;
        }
    }
    
    // Set output tensors
    for (uint32_t i = 0; i < output_count; ++i) {
        cmx_status status = set_output(i, &outputs[i]);
        if (status != cmx_status::SUCCESS) {
            return status;
        }
    }
    
    return run();
}

// Batch execution
cmx_status cmx_graph_executor::run_batch(cmx_tensor** input_batches, uint32_t* input_counts,
                                        cmx_tensor** output_batches, uint32_t* output_counts,
                                        uint32_t batch_size) {
    if (batch_size > config_.max_batch_size) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        cmx_status status = run(input_batches[batch], input_counts[batch],
                               output_batches[batch], output_counts[batch]);
        if (status != cmx_status::SUCCESS) {
            return status;
        }
    }
    
    return cmx_status::SUCCESS;
}

// Input/Output management
cmx_status cmx_graph_executor::set_input(uint32_t index, cmx_tensor* tensor) {
    if (!graph_ || index >= graph_->input_count || !tensor) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    uint32_t node_id = graph_->input_nodes[index];
    cmx_graph_node* node = &graph_->nodes[node_id];
    
    if (node->context) {
        return cmx_set_input(node->context, 0, tensor);
    }
    
    return cmx_status::ERROR_INVALID_CONTEXT;
}

cmx_status cmx_graph_executor::set_output(uint32_t index, cmx_tensor* tensor) {
    if (!graph_ || index >= graph_->output_count || !tensor) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    uint32_t node_id = graph_->output_nodes[index];
    cmx_graph_node* node = &graph_->nodes[node_id];
    
    if (node->context) {
        return cmx_set_output(node->context, 0, tensor);
    }
    
    return cmx_status::ERROR_INVALID_CONTEXT;
}

cmx_tensor* cmx_graph_executor::get_input(uint32_t index) {
    if (!graph_ || index >= graph_->input_count) {
        return nullptr;
    }
    
    uint32_t node_id = graph_->input_nodes[index];
    cmx_graph_node* node = &graph_->nodes[node_id];
    
    if (node->context) {
        return cmx_get_input(node->context, 0);
    }
    
    return nullptr;
}

cmx_tensor* cmx_graph_executor::get_output(uint32_t index) {
    if (!graph_ || index >= graph_->output_count) {
        return nullptr;
    }
    
    uint32_t node_id = graph_->output_nodes[index];
    cmx_graph_node* node = &graph_->nodes[node_id];
    
    if (node->context) {
        return cmx_get_output(node->context, 0);
    }
    
    return nullptr;
}

// Graph inspection
uint32_t cmx_graph_executor::get_input_count() const {
    return graph_ ? graph_->input_count : 0;
}

uint32_t cmx_graph_executor::get_output_count() const {
    return graph_ ? graph_->output_count : 0;
}

uint32_t cmx_graph_executor::get_node_count() const {
    return graph_ ? graph_->node_count : 0;
}

const cmx_graph_node* cmx_graph_executor::get_node(uint32_t index) const {
    if (!graph_ || index >= graph_->node_count) {
        return nullptr;
    }
    return &graph_->nodes[index];
}

// Configuration
void cmx_graph_executor::set_config(const cmx_graph_config& config) {
    config_ = config;
    if (executor_) {
        executor_->set_config(config_.executor_config);
    }
}

const cmx_graph_config& cmx_graph_executor::get_config() const {
    return config_;
}

// Statistics and profiling
const cmx_graph_stats& cmx_graph_executor::get_stats() const {
    return stats_;
}

void cmx_graph_executor::reset_stats() {
    std::memset(&stats_, 0, sizeof(stats_));
    peak_memory_usage_ = 0;
}

cmx_status cmx_graph_executor::get_node_profile(uint32_t node_id, uint64_t& execution_time) {
    if (!graph_ || node_id >= graph_->node_count) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    const cmx_graph_node* node = &graph_->nodes[node_id];
    if (node->context) {
        execution_time = cmx_get_execution_time(node->context);
        return cmx_status::SUCCESS;
    }
    
    return cmx_status::ERROR_INVALID_CONTEXT;
}

// Memory management
size_t cmx_graph_executor::get_memory_usage() const {
    return current_memory_usage_;
}

cmx_status cmx_graph_executor::optimize_memory() {
    if (!graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Simple memory optimization - reuse tensors where possible
    return optimize_memory_layout();
}

// Private methods
cmx_status cmx_graph_executor::initialize_graph() {
    if (!graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Initialize contexts for all nodes
    for (uint32_t i = 0; i < graph_->node_count; ++i) {
        cmx_graph_node* node = &graph_->nodes[i];
        if (!node->context) {
            node->context = cmx_create_op_context();
            if (!node->context) {
                return cmx_status::ERROR_OUT_OF_MEMORY;
            }
            node->owns_context = true;
        }
        
        // Set node properties
        node->is_ready = false;
        node->is_executed = false;
    }
    
    return allocate_tensors();
}

cmx_status cmx_graph_executor::schedule_execution() {
    return topological_sort();
}

cmx_status cmx_graph_executor::execute_node(cmx_graph_node* node) {
    if (!node || !node->context) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    // Check if node is ready for execution
    if (!is_node_ready(node)) {
        return cmx_status::ERROR_EXECUTION_FAILED;
    }
    
    // Execute the operation
    cmx_status status = executor_->execute_op(node->op_name, *node->context);
    
    if (status == cmx_status::SUCCESS) {
        mark_node_executed(node);
    }
    
    return status;
}

cmx_status cmx_graph_executor::validate_graph() {
    if (!graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Basic validation - check that all nodes have valid operations
    for (uint32_t i = 0; i < graph_->node_count; ++i) {
        const cmx_graph_node* node = &graph_->nodes[i];
        if (!cmx_is_op_registered(node->op_name)) {
            return cmx_status::ERROR_UNSUPPORTED_OP;
        }
    }
    
    return cmx_status::SUCCESS;
}

// Optimization methods (simplified implementations)
cmx_status cmx_graph_executor::optimize_execution_order() {
    return topological_sort();
}

cmx_status cmx_graph_executor::optimize_memory_layout() {
    // Placeholder for memory layout optimization
    return cmx_status::SUCCESS;
}

cmx_status cmx_graph_executor::fuse_operations() {
    // Placeholder for operation fusion
    return cmx_status::SUCCESS;
}

// Scheduling methods
cmx_status cmx_graph_executor::topological_sort() {
    if (!graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Simple topological sort implementation
    // In practice, this would be more sophisticated
    for (uint32_t i = 0; i < graph_->node_count; ++i) {
        graph_->execution_order[i] = i;
        graph_->nodes[i].execution_order = i;
    }
    
    is_optimized_ = true;
    return cmx_status::SUCCESS;
}

bool cmx_graph_executor::is_node_ready(const cmx_graph_node* node) {
    // Check if all input nodes have been executed
    for (uint32_t i = 0; i < node->input_count; ++i) {
        uint32_t input_node_id = node->input_nodes[i];
        if (!graph_->nodes[input_node_id].is_executed) {
            return false;
        }
    }
    return true;
}

void cmx_graph_executor::mark_node_executed(cmx_graph_node* node) {
    node->is_executed = true;
}

// Memory management helpers
cmx_status cmx_graph_executor::allocate_tensors() {
    if (!graph_) {
        return cmx_status::ERROR_INVALID_CONTEXT;
    }
    
    // Calculate total memory needed
    size_t total_memory = 0;
    for (uint32_t i = 0; i < graph_->tensor_count; ++i) {
        total_memory += graph_->tensors[i].byte_size;
    }
    
    if (config_.memory_limit > 0 && total_memory > config_.memory_limit) {
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    current_memory_usage_ = total_memory;
    return cmx_status::SUCCESS;
}

void cmx_graph_executor::free_tensors() {
    // Placeholder for tensor deallocation
}

void cmx_graph_executor::update_memory_usage() {
    // Placeholder for dynamic memory usage tracking
}

// C-style API functions
cmx_graph_executor* cmx_create_graph_executor() {
    return new cmx_graph_executor();
}

cmx_graph_executor* cmx_create_graph_executor_with_config(const cmx_graph_config& config) {
    return new cmx_graph_executor(config);
}

void cmx_destroy_graph_executor(cmx_graph_executor* executor) {
    delete executor;
}

cmx_status cmx_graph_load_model(cmx_graph_executor* executor, const cmx_model& model) {
    if (!executor) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    return executor->load(model);
}

cmx_status cmx_graph_run(cmx_graph_executor* executor) {
    if (!executor) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    return executor->run();
}

cmx_status cmx_graph_run_with_io(cmx_graph_executor* executor,
                                cmx_tensor* inputs, uint32_t input_count,
                                cmx_tensor* outputs, uint32_t output_count) {
    if (!executor) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    return executor->run(inputs, input_count, outputs, output_count);
}

// Utility functions
cmx_graph_config cmx_default_graph_config() {
    cmx_graph_config config;
    config.enable_profiling = false;
    config.enable_optimization = true;
    config.enable_memory_reuse = true;
    config.enable_parallel_execution = false;
    config.max_batch_size = 1;
    config.memory_limit = 0; // No limit
    config.executor_config = cmx_default_executor_config();
    return config;
}

cmx_status cmx_create_graph(cmx_graph** graph, uint32_t max_nodes, uint32_t max_tensors) {
    if (!graph) {
        return cmx_status::ERROR_INVALID_ARGS;
    }
    
    *graph = static_cast<cmx_graph*>(std::calloc(1, sizeof(cmx_graph)));
    if (!*graph) {
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    cmx_graph* g = *graph;
    g->max_nodes = max_nodes;
    g->max_tensors = max_tensors;
    
    // Allocate arrays
    g->nodes = static_cast<cmx_graph_node*>(std::calloc(max_nodes, sizeof(cmx_graph_node)));
    g->tensors = static_cast<cmx_tensor*>(std::calloc(max_tensors, sizeof(cmx_tensor)));
    g->execution_order = static_cast<uint32_t*>(std::calloc(max_nodes, sizeof(uint32_t)));
    g->input_nodes = static_cast<uint32_t*>(std::calloc(max_nodes, sizeof(uint32_t)));
    g->output_nodes = static_cast<uint32_t*>(std::calloc(max_nodes, sizeof(uint32_t)));
    
    if (!g->nodes || !g->tensors || !g->execution_order || !g->input_nodes || !g->output_nodes) {
        cmx_destroy_graph(g);
        *graph = nullptr;
        return cmx_status::ERROR_OUT_OF_MEMORY;
    }
    
    g->owns_tensors = true;
    g->owns_nodes = true;
    
    return cmx_status::SUCCESS;
}

void cmx_destroy_graph(cmx_graph* graph) {
    if (!graph) return;
    
    // Free node contexts
    if (graph->nodes) {
        for (uint32_t i = 0; i < graph->node_count; ++i) {
            if (graph->nodes[i].owns_context && graph->nodes[i].context) {
                cmx_destroy_op_context(graph->nodes[i].context);
            }
        }
        std::free(graph->nodes);
    }
    
    if (graph->tensors) std::free(graph->tensors);
    if (graph->execution_order) std::free(graph->execution_order);
    if (graph->input_nodes) std::free(graph->input_nodes);
    if (graph->output_nodes) std::free(graph->output_nodes);
    
    std::free(graph);
}

} // namespace cmx