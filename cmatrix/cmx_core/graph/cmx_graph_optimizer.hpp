#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace cmx {
namespace graph {

// Forward declarations
class CMXGraph;
class CMXNode;

/// @brief Optimization pass result
enum class OptimizationResult {
    SUCCESS,
    FAILED,
    NO_CHANGES,
    GRAPH_INVALID
};

/// @brief Optimization statistics
struct OptimizationStats {
    uint32_t nodes_removed;
    uint32_t nodes_fused;
    uint32_t constants_folded;
    uint32_t memory_saved_bytes;
    uint32_t operations_simplified;
    float optimization_time_ms;
    
    OptimizationStats() : nodes_removed(0), nodes_fused(0), constants_folded(0),
                          memory_saved_bytes(0), operations_simplified(0), optimization_time_ms(0.0f) {}
    
    /// @brief Add stats from another optimization pass
    void accumulate(const OptimizationStats& other) {
        nodes_removed += other.nodes_removed;
        nodes_fused += other.nodes_fused;
        constants_folded += other.constants_folded;
        memory_saved_bytes += other.memory_saved_bytes;
        operations_simplified += other.operations_simplified;
        optimization_time_ms += other.optimization_time_ms;
    }
    
    /// @brief Check if any optimizations were applied
    bool has_changes() const {
        return nodes_removed > 0 || nodes_fused > 0 || constants_folded > 0 || operations_simplified > 0;
    }
};

/// @brief Optimization pass function signature
using OptimizationPass = std::function<OptimizationResult(CMXGraph&, OptimizationStats&)>;

/// @brief Optimization pass configuration
struct OptimizationPassConfig {
    bool enabled;
    uint32_t max_iterations;
    float convergence_threshold;
    
    OptimizationPassConfig() : enabled(true), max_iterations(1), convergence_threshold(0.01f) {}
};

/// @brief Graph optimization engine
class CMXGraphOptimizer {
public:
    CMXGraphOptimizer();
    ~CMXGraphOptimizer() = default;
    
    /// @brief Run all optimization passes on graph
    /// @param graph Graph to optimize
    /// @param stats Output optimization statistics
    /// @return Optimization result
    OptimizationResult optimize(CMXGraph& graph, OptimizationStats& stats);
    
    /// @brief Run specific optimization pass
    /// @param graph Graph to optimize
    /// @param pass_name Name of optimization pass
    /// @param stats Output optimization statistics
    /// @return Optimization result
    OptimizationResult run_pass(CMXGraph& graph, const std::string& pass_name, OptimizationStats& stats);
    
    /// @brief Register custom optimization pass
    /// @param name Pass name
    /// @param pass Optimization pass function
    /// @param config Pass configuration
    void register_pass(const std::string& name, OptimizationPass pass, const OptimizationPassConfig& config);
    
    /// @brief Enable/disable optimization pass
    /// @param name Pass name
    /// @param enabled Whether to enable the pass
    void set_pass_enabled(const std::string& name, bool enabled);
    
    /// @brief Set maximum iterations for convergence-based passes
    /// @param name Pass name
    /// @param max_iterations Maximum iterations
    void set_pass_max_iterations(const std::string& name, uint32_t max_iterations);
    
    /// @brief Get list of registered passes
    /// @return Vector of pass names
    std::vector<std::string> get_pass_names() const;
    
    /// @brief Clear all optimization statistics
    void clear_stats();
    
    /// @brief Get cumulative optimization statistics
    /// @return Optimization statistics
    const OptimizationStats& get_cumulative_stats() const { return cumulative_stats_; }
    
private:
    struct PassInfo {
        OptimizationPass pass;
        OptimizationPassConfig config;
        
        PassInfo(OptimizationPass p, const OptimizationPassConfig& c) : pass(std::move(p)), config(c) {}
    };
    
    std::vector<std::pair<std::string, PassInfo>> passes_;
    std::unordered_map<std::string, size_t> pass_index_map_;
    OptimizationStats cumulative_stats_;
    
    /// @brief Register built-in optimization passes
    void register_builtin_passes();
    
    /// @brief Built-in optimization passes
    static OptimizationResult constant_folding_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult dead_code_elimination_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult operator_fusion_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult algebraic_simplification_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult memory_layout_optimization_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult redundant_transpose_elimination_pass(CMXGraph& graph, OptimizationStats& stats);
    static OptimizationResult batch_normalization_folding_pass(CMXGraph& graph, OptimizationStats& stats);
    
    /// @brief Helper functions
    static bool is_constant_node(const CMXNode& node);
    static bool is_dead_node(const CMXNode& node, const CMXGraph& graph);
    static bool can_fuse_nodes(const CMXNode& producer, const CMXNode& consumer);
    static bool are_nodes_fusable(const std::string& op1, const std::string& op2);
    static bool is_identity_operation(const CMXNode& node);
    static bool has_single_consumer(const CMXNode& node, const CMXGraph& graph);
    
    /// @brief Validation helpers
    static bool validate_graph_integrity(const CMXGraph& graph);
    static bool validate_node_connections(const CMXGraph& graph);
};

/// @brief Optimization pass builder for custom passes
class OptimizationPassBuilder {
public:
    OptimizationPassBuilder() = default;
    
    /// @brief Create constant folding pass
    /// @return Optimization pass function
    static OptimizationPass create_constant_folding_pass();
    
    /// @brief Create dead code elimination pass
    /// @return Optimization pass function
    static OptimizationPass create_dce_pass();
    
    /// @brief Create operator fusion pass
    /// @param fusion_patterns Map of fusion patterns
    /// @return Optimization pass function
    static OptimizationPass create_fusion_pass(const std::unordered_map<std::string, std::string>& fusion_patterns);
    
    /// @brief Create algebraic simplification pass
    /// @return Optimization pass function
    static OptimizationPass create_algebraic_simplification_pass();
    
    /// @brief Create memory optimization pass
    /// @return Optimization pass function
    static OptimizationPass create_memory_optimization_pass();
};

} // namespace graph
} // namespace cmx