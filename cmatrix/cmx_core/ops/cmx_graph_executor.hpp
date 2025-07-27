#ifndef CMX_GRAPH_EXECUTOR_HPP
#define CMX_GRAPH_EXECUTOR_HPP

#include "cmx_ops.hpp"
#include "cmx_op_context.hpp"
#include "cmx_op_executor.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace cmx {

// Forward declarations
struct cmx_model;
struct cmx_graph_node;

// Graph execution statistics
struct cmx_graph_stats {
    uint64_t total_graphs_executed;
    uint64_t total_nodes_executed;
    uint64_t total_execution_time;
    uint64_t avg_graph_execution_time;
    uint32_t memory_peak_usage;
    uint32_t failed_executions;
};

// Graph execution configuration
struct cmx_graph_config {
    bool enable_profiling;
    bool enable_optimization;
    bool enable_memory_reuse;
    bool enable_parallel_execution;
    uint32_t max_batch_size;
    size_t memory_limit;
    cmx_executor_config executor_config;
};

// Graph node representation
struct cmx_graph_node {
    uint32_t node_id;
    std::string op_name;
    cmx_op_type op_type;
    
    // Node connections
    uint32_t* input_nodes;
    uint32_t* output_nodes;
    uint32_t input_count;
    uint32_t output_count;
    
    // Tensor information
    uint32_t* input_tensor_ids;
    uint32_t* output_tensor_ids;
    
    // Execution context
    cmx_op_context* context;
    
    // Scheduling information
    uint32_t execution_order;
    bool is_ready;
    bool is_executed;
    
    // Memory management
    bool owns_context;
};

// Computation graph representation
struct cmx_graph {
    cmx_graph_node* nodes;
    uint32_t node_count;
    uint32_t max_nodes;
    
    // Graph topology
    uint32_t* execution_order;
    uint32_t* input_nodes;
    uint32_t* output_nodes;
    uint32_t input_count;
    uint32_t output_count;
    
    // Tensor management
    cmx_tensor* tensors;
    uint32_t tensor_count;
    uint32_t max_tensors;
    
    // Memory management
    bool owns_tensors;
    bool owns_nodes;
};

// Graph executor class
class cmx_graph_executor {
public:
    cmx_graph_executor();
    explicit cmx_graph_executor(const cmx_graph_config& config);
    ~cmx_graph_executor();
    
    // Model loading
    cmx_status load(const cmx_model& model);
    cmx_status load_from_graph(const cmx_graph& graph);
    
    // Graph execution
    cmx_status run();
    cmx_status run(cmx_tensor* inputs, uint32_t input_count,
                   cmx_tensor* outputs, uint32_t output_count);
    
    // Batch execution
    cmx_status run_batch(cmx_tensor** input_batches, uint32_t* input_counts,
                        cmx_tensor** output_batches, uint32_t* output_counts,
                        uint32_t batch_size);
    
    // Input/Output management
    cmx_status set_input(uint32_t index, cmx_tensor* tensor);
    cmx_status set_output(uint32_t index, cmx_tensor* tensor);
    cmx_tensor* get_input(uint32_t index);
    cmx_tensor* get_output(uint32_t index);
    
    // Graph inspection
    uint32_t get_input_count() const;
    uint32_t get_output_count() const;
    uint32_t get_node_count() const;
    const cmx_graph_node* get_node(uint32_t index) const;
    
    // Configuration
    void set_config(const cmx_graph_config& config);
    const cmx_graph_config& get_config() const;
    
    // Statistics and profiling
    const cmx_graph_stats& get_stats() const;
    void reset_stats();
    cmx_status get_node_profile(uint32_t node_id, uint64_t& execution_time);
    
    // Memory management
    size_t get_memory_usage() const;
    cmx_status optimize_memory();
    
private:
    cmx_graph_config config_;
    cmx_graph_stats stats_;
    cmx_graph* graph_;
    cmx_op_executor* executor_;
    
    // Execution state
    bool is_loaded_;
    bool is_optimized_;
    
    // Memory management
    size_t current_memory_usage_;
    size_t peak_memory_usage_;
    
    // Internal methods
    cmx_status initialize_graph();
    cmx_status schedule_execution();
    cmx_status execute_node(cmx_graph_node* node);
    cmx_status validate_graph();
    
    // Optimization methods
    cmx_status optimize_execution_order();
    cmx_status optimize_memory_layout();
    cmx_status fuse_operations();
    
    // Scheduling methods
    cmx_status topological_sort();
    bool is_node_ready(const cmx_graph_node* node);
    void mark_node_executed(cmx_graph_node* node);
    
    // Memory management helpers
    cmx_status allocate_tensors();
    void free_tensors();
    void update_memory_usage();
};

// C-style API functions
cmx_graph_executor* cmx_create_graph_executor();
cmx_graph_executor* cmx_create_graph_executor_with_config(const cmx_graph_config& config);
void cmx_destroy_graph_executor(cmx_graph_executor* executor);

cmx_status cmx_graph_load_model(cmx_graph_executor* executor, const cmx_model& model);
cmx_status cmx_graph_run(cmx_graph_executor* executor);
cmx_status cmx_graph_run_with_io(cmx_graph_executor* executor,
                                cmx_tensor* inputs, uint32_t input_count,
                                cmx_tensor* outputs, uint32_t output_count);

// Utility functions
cmx_graph_config cmx_default_graph_config();
cmx_status cmx_create_graph(cmx_graph** graph, uint32_t max_nodes, uint32_t max_tensors);
void cmx_destroy_graph(cmx_graph* graph);

// Graph building helpers
cmx_status cmx_add_graph_node(cmx_graph* graph, const std::string& op_name,
                             uint32_t* input_tensor_ids, uint32_t input_count,
                             uint32_t* output_tensor_ids, uint32_t output_count);
cmx_status cmx_connect_nodes(cmx_graph* graph, uint32_t from_node, uint32_t to_node);
cmx_status cmx_add_tensor(cmx_graph* graph, const cmx_tensor& tensor, uint32_t* tensor_id);

} // namespace cmx

#endif // CMX_GRAPH_EXECUTOR_HPP

