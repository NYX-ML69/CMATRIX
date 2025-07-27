 #pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>

namespace cmx {
namespace graph {

class CMXNode;
class CMXTensor;

using NodeID = uint32_t;
using TensorID = uint32_t;
using NodePtr = std::shared_ptr<CMXNode>;

/**
 * @brief Main computation graph representation
 * 
 * Manages nodes, tensors, and their relationships in a computation graph.
 * Provides methods for graph construction, traversal, and manipulation.
 */
class CMXGraph {
public:
    CMXGraph();
    ~CMXGraph();

    // Graph construction
    /**
     * @brief Add a node to the graph
     * @param node Node to add
     * @return NodeID of the added node
     */
    NodeID add_node(NodePtr node);

    /**
     * @brief Get node by ID
     * @param node_id ID of the node to retrieve
     * @return Pointer to the node, nullptr if not found
     */
    NodePtr get_node(NodeID node_id) const;

    /**
     * @brief Get all input nodes (nodes with no predecessors)
     * @return Vector of input node IDs
     */
    std::vector<NodeID> get_input_nodes() const;

    /**
     * @brief Get all output nodes (nodes with no successors)
     * @return Vector of output node IDs
     */
    std::vector<NodeID> get_output_nodes() const;

    // Graph traversal
    /**
     * @brief Perform topological sort of nodes
     * @return Vector of node IDs in topological order
     */
    std::vector<NodeID> topological_sort() const;

    /**
     * @brief Get direct predecessors of a node
     * @param node_id ID of the node
     * @return Vector of predecessor node IDs
     */
    std::vector<NodeID> get_predecessors(NodeID node_id) const;

    /**
     * @brief Get direct successors of a node
     * @param node_id ID of the node
     * @return Vector of successor node IDs
     */
    std::vector<NodeID> get_successors(NodeID node_id) const;

    // Graph manipulation
    /**
     * @brief Remove a node from the graph
     * @param node_id ID of the node to remove
     * @return true if node was removed, false if not found
     */
    bool remove_node(NodeID node_id);

    /**
     * @brief Create a deep copy of the graph
     * @return New graph instance with copied nodes
     */
    std::unique_ptr<CMXGraph> clone() const;

    /**
     * @brief Extract subgraph containing specified nodes
     * @param node_ids Vector of node IDs to include
     * @return New graph containing only specified nodes and their connections
     */
    std::unique_ptr<CMXGraph> extract_subgraph(const std::vector<NodeID>& node_ids) const;

    // Graph properties
    /**
     * @brief Get total number of nodes in the graph
     * @return Number of nodes
     */
    size_t node_count() const;

    /**
     * @brief Check if graph is empty
     * @return true if no nodes, false otherwise
     */
    bool empty() const;

    /**
     * @brief Validate graph structure
     * @return true if graph is valid, false otherwise
     */
    bool validate() const;

    // Tensor management
    /**
     * @brief Register a tensor with the graph
     * @param tensor_id Unique tensor identifier
     * @param tensor Tensor object
     */
    void register_tensor(TensorID tensor_id, std::shared_ptr<CMXTensor> tensor);

    /**
     * @brief Get tensor by ID
     * @param tensor_id ID of the tensor
     * @return Pointer to tensor, nullptr if not found
     */
    std::shared_ptr<CMXTensor> get_tensor(TensorID tensor_id) const;

    /**
     * @brief Get all registered tensor IDs
     * @return Vector of tensor IDs
     */
    std::vector<TensorID> get_tensor_ids() const;

    // Debug and utility
    /**
     * @brief Generate string representation of the graph
     * @return String describing graph structure
     */
    std::string to_string() const;

    /**
     * @brief Get graph statistics
     * @return Map of statistic names to values
     */
    std::unordered_map<std::string, uint32_t> get_stats() const;

private:
    // Internal data structures
    std::unordered_map<NodeID, NodePtr> nodes_;
    std::unordered_map<NodeID, std::vector<NodeID>> adjacency_list_;
    std::unordered_map<NodeID, std::vector<NodeID>> reverse_adjacency_list_;
    std::unordered_map<TensorID, std::shared_ptr<CMXTensor>> tensors_;
    
    NodeID next_node_id_;

    // Internal helper methods
    void update_adjacency_lists();
    bool has_cycle_util(NodeID node_id, std::unordered_map<NodeID, int>& color) const;
    void clone_node_recursive(NodeID node_id, const CMXGraph& source, 
                             std::unordered_map<NodeID, NodeID>& node_mapping) const;
};

} // namespace graph
} // namespace cmx