#pragma once

#include "cmx_graph_executor.hpp"
#include "cmx_op_loader.hpp"
#include "cmx_error.hpp"
#include "cmx_types.hpp"

namespace cmx {

/**
 * @brief Graph structure representing a computational graph
 */
struct cmx_graph {
    void* internal_graph;  // Opaque pointer to internal implementation
    uint32_t node_count;
    uint32_t edge_count;
    cmx_status last_status;
};

/**
 * @brief Node structure representing a single operation in the graph
 */
struct cmx_node {
    uint32_t id;
    const char* op_type;
    void* op_params;
    uint32_t input_count;
    uint32_t output_count;
};

/**
 * @brief Create a new computational graph
 * @return Initialized graph structure
 */
cmx_graph cmx_create_graph();

/**
 * @brief Destroy a graph and free associated resources
 * @param graph Pointer to graph to destroy
 * @return Status code indicating success or failure
 */
cmx_status cmx_destroy_graph(cmx_graph* graph);

/**
 * @brief Add a node to the computational graph
 * @param graph Reference to the target graph
 * @param op Operation to add as a node
 * @return Status code indicating success or failure
 */
cmx_status cmx_add_node(cmx_graph& graph, const cmx_op& op);

/**
 * @brief Add an edge between two nodes in the graph
 * @param graph Reference to the target graph
 * @param src_node_id Source node identifier
 * @param dst_node_id Destination node identifier
 * @param port_id Port identifier for the connection
 * @return Status code indicating success or failure
 */
cmx_status cmx_add_edge(cmx_graph& graph, uint32_t src_node_id, uint32_t dst_node_id, uint32_t port_id);

/**
 * @brief Execute the computational graph
 * @param graph Reference to the graph to execute
 * @return Status code indicating success or failure
 */
cmx_status cmx_run_graph(cmx_graph& graph);

/**
 * @brief Execute the graph with input tensors
 * @param graph Reference to the graph to execute
 * @param inputs Array of input tensors
 * @param input_count Number of input tensors
 * @param outputs Array to store output tensors
 * @param output_count Number of expected outputs
 * @return Status code indicating success or failure
 */
cmx_status cmx_run_graph_with_tensors(cmx_graph& graph, 
                                      const cmx_tensor* inputs, uint32_t input_count,
                                      cmx_tensor* outputs, uint32_t output_count);

/**
 * @brief Optimize the graph for better performance
 * @param graph Reference to the graph to optimize
 * @param optimization_level Level of optimization (0-3)
 * @return Status code indicating success or failure
 */
cmx_status cmx_optimize_graph(cmx_graph& graph, uint32_t optimization_level);

/**
 * @brief Validate graph structure and operations
 * @param graph Reference to the graph to validate
 * @return Status code indicating success or failure
 */
cmx_status cmx_validate_graph(const cmx_graph& graph);

/**
 * @brief Get graph execution statistics
 * @param graph Reference to the graph
 * @param stats Pointer to structure to fill with statistics
 * @return Status code indicating success or failure
 */
cmx_status cmx_get_graph_stats(const cmx_graph& graph, cmx_graph_stats* stats);

} // namespace cmx