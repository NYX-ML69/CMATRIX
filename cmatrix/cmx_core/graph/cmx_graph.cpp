#include "cmx_graph.hpp"
#include "cmx_node.hpp"
#include <algorithm>
#include <sstream>
#include <stack>
#include <queue>

namespace cmx {
namespace graph {

CMXGraph::CMXGraph() : next_node_id_(1) {}

CMXGraph::~CMXGraph() {
    nodes_.clear();
    adjacency_list_.clear();
    reverse_adjacency_list_.clear();
    tensors_.clear();
}

NodeID CMXGraph::add_node(NodePtr node) {
    if (!node) {
        return 0; // Invalid node ID
    }
    
    NodeID node_id = next_node_id_++;
    nodes_[node_id] = node;
    
    // Initialize adjacency lists
    adjacency_list_[node_id] = std::vector<NodeID>();
    reverse_adjacency_list_[node_id] = std::vector<NodeID>();
    
    update_adjacency_lists();
    return node_id;
}

NodePtr CMXGraph::get_node(NodeID node_id) const {
    auto it = nodes_.find(node_id);
    return (it != nodes_.end()) ? it->second : nullptr;
}

std::vector<NodeID> CMXGraph::get_input_nodes() const {
    std::vector<NodeID> input_nodes;
    for (const auto& [node_id, _] : nodes_) {
        if (reverse_adjacency_list_.at(node_id).empty()) {
            input_nodes.push_back(node_id);
        }
    }
    return input_nodes;
}

std::vector<NodeID> CMXGraph::get_output_nodes() const {
    std::vector<NodeID> output_nodes;
    for (const auto& [node_id, _] : nodes_) {
        if (adjacency_list_.at(node_id).empty()) {
            output_nodes.push_back(node_id);
        }
    }
    return output_nodes;
}

std::vector<NodeID> CMXGraph::topological_sort() const {
    std::vector<NodeID> result;
    std::unordered_map<NodeID, int> in_degree;
    
    // Calculate in-degrees
    for (const auto& [node_id, _] : nodes_) {
        in_degree[node_id] = reverse_adjacency_list_.at(node_id).size();
    }
    
    // Initialize queue with nodes having in-degree 0
    std::queue<NodeID> queue;
    for (const auto& [node_id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(node_id);
        }
    }
    
    // Process nodes
    while (!queue.empty()) {
        NodeID current = queue.front();
        queue.pop();
        result.push_back(current);
        
        // Update in-degrees of successors
        for (NodeID successor : adjacency_list_.at(current)) {
            in_degree[successor]--;
            if (in_degree[successor] == 0) {
                queue.push(successor);
            }
        }
    }
    
    // Check for cycles
    if (result.size() != nodes_.size()) {
        result.clear(); // Return empty vector if cycle detected
    }
    
    return result;
}

std::vector<NodeID> CMXGraph::get_predecessors(NodeID node_id) const {
    auto it = reverse_adjacency_list_.find(node_id);
    return (it != reverse_adjacency_list_.end()) ? it->second : std::vector<NodeID>();
}

std::vector<NodeID> CMXGraph::get_successors(NodeID node_id) const {
    auto it = adjacency_list_.find(node_id);
    return (it != adjacency_list_.end()) ? it->second : std::vector<NodeID>();
}

bool CMXGraph::remove_node(NodeID node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return false;
    }
    
    // Remove from adjacency lists
    adjacency_list_.erase(node_id);
    reverse_adjacency_list_.erase(node_id);
    
    // Remove references from other nodes
    for (auto& [_, successors] : adjacency_list_) {
        successors.erase(std::remove(successors.begin(), successors.end(), node_id), 
                        successors.end());
    }
    
    for (auto& [_, predecessors] : reverse_adjacency_list_) {
        predecessors.erase(std::remove(predecessors.begin(), predecessors.end(), node_id), 
                          predecessors.end());
    }
    
    nodes_.erase(it);
    return true;
}

std::unique_ptr<CMXGraph> CMXGraph::clone() const {
    auto cloned_graph = std::make_unique<CMXGraph>();
    std::unordered_map<NodeID, NodeID> node_mapping;
    
    // Clone all nodes
    for (const auto& [node_id, node] : nodes_) {
        auto cloned_node = std::make_shared<CMXNode>(*node);
        NodeID new_node_id = cloned_graph->add_node(cloned_node);
        node_mapping[node_id] = new_node_id;
    }
    
    // Clone tensors
    for (const auto& [tensor_id, tensor] : tensors_) {
        cloned_graph->register_tensor(tensor_id, tensor);
    }
    
    return cloned_graph;
}

std::unique_ptr<CMXGraph> CMXGraph::extract_subgraph(const std::vector<NodeID>& node_ids) const {
    auto subgraph = std::make_unique<CMXGraph>();
    std::unordered_map<NodeID, NodeID> node_mapping;
    
    // Add specified nodes to subgraph
    for (NodeID node_id : node_ids) {
        auto it = nodes_.find(node_id);
        if (it != nodes_.end()) {
            auto cloned_node = std::make_shared<CMXNode>(*it->second);
            NodeID new_node_id = subgraph->add_node(cloned_node);
            node_mapping[node_id] = new_node_id;
        }
    }
    
    return subgraph;
}

size_t CMXGraph::node_count() const {
    return nodes_.size();
}

bool CMXGraph::empty() const {
    return nodes_.empty();
}

bool CMXGraph::validate() const {
    // Check for cycles
    std::unordered_map<NodeID, int> color; // 0: white, 1: gray, 2: black
    
    for (const auto& [node_id, _] : nodes_) {
        color[node_id] = 0;
    }
    
    for (const auto& [node_id, _] : nodes_) {
        if (color[node_id] == 0) {
            if (has_cycle_util(node_id, color)) {
                return false;
            }
        }
    }
    
    return true;
}

void CMXGraph::register_tensor(TensorID tensor_id, std::shared_ptr<CMXTensor> tensor) {
    tensors_[tensor_id] = tensor;
}

std::shared_ptr<CMXTensor> CMXGraph::get_tensor(TensorID tensor_id) const {
    auto it = tensors_.find(tensor_id);
    return (it != tensors_.end()) ? it->second : nullptr;
}

std::vector<TensorID> CMXGraph::get_tensor_ids() const {
    std::vector<TensorID> tensor_ids;
    for (const auto& [tensor_id, _] : tensors_) {
        tensor_ids.push_back(tensor_id);
    }
    return tensor_ids;
}

std::string CMXGraph::to_string() const {
    std::ostringstream oss;
    oss << "CMXGraph {\n";
    oss << "  Nodes: " << nodes_.size() << "\n";
    oss << "  Tensors: " << tensors_.size() << "\n";
    
    for (const auto& [node_id, node] : nodes_) {
        oss << "  Node " << node_id << ": " << node->to_string() << "\n";
    }
    
    oss << "}";
    return oss.str();
}

std::unordered_map<std::string, uint32_t> CMXGraph::get_stats() const {
    std::unordered_map<std::string, uint32_t> stats;
    stats["node_count"] = static_cast<uint32_t>(nodes_.size());
    stats["tensor_count"] = static_cast<uint32_t>(tensors_.size());
    stats["input_nodes"] = static_cast<uint32_t>(get_input_nodes().size());
    stats["output_nodes"] = static_cast<uint32_t>(get_output_nodes().size());
    
    // Count edges
    uint32_t edge_count = 0;
    for (const auto& [_, successors] : adjacency_list_) {
        edge_count += static_cast<uint32_t>(successors.size());
    }
    stats["edge_count"] = edge_count;
    
    return stats;
}

void CMXGraph::update_adjacency_lists() {
    // This method would be implemented based on how nodes store their connections
    // For now, it's a placeholder that would analyze node input/output relationships
    // and update the adjacency lists accordingly
}

bool CMXGraph::has_cycle_util(NodeID node_id, std::unordered_map<NodeID, int>& color) const {
    color[node_id] = 1; // Gray
    
    for (NodeID successor : adjacency_list_.at(node_id)) {
        if (color[successor] == 1) {
            return true; // Back edge found
        }
        if (color[successor] == 0 && has_cycle_util(successor, color)) {
            return true;
        }
    }
    
    color[node_id] = 2; // Black
    return false;
}

void CMXGraph::clone_node_recursive(NodeID node_id, const CMXGraph& source,
                                   std::unordered_map<NodeID, NodeID>& node_mapping) const {
    // Implementation for recursive node cloning
    // This would be used in more complex cloning scenarios
}

} // namespace graph
} // namespace cmx