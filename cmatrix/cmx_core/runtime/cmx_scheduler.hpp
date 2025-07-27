#pragma once

#include <functional>
#include <atomic>
#include <cstdint>
#include <mutex>

namespace cmx {
namespace runtime {

/**
 * @brief Task priority levels for scheduler
 */
enum class TaskPriority : uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Task execution status
 */
enum class TaskStatus : uint8_t {
    PENDING = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3
};

/**
 * @brief Lightweight task abstraction for operator execution
 */
struct Task {
    uint32_t id;                           ///< Unique task identifier
    std::function<void()> function_ptr;    ///< Function to execute
    TaskPriority priority;                 ///< Task priority level
    TaskStatus status;                     ///< Current execution status
    uint32_t dependency_count;             ///< Number of dependencies remaining
    Task* dependencies[8];                 ///< Fixed-size dependency array
    uint8_t dep_index;                     ///< Current dependency index
    
    /**
     * @brief Default constructor
     */
    Task() : id(0), priority(TaskPriority::NORMAL), status(TaskStatus::PENDING), 
             dependency_count(0), dep_index(0) {
        for (int i = 0; i < 8; ++i) {
            dependencies[i] = nullptr;
        }
    }
    
    /**
     * @brief Constructor with parameters
     */
    Task(uint32_t task_id, std::function<void()> func, TaskPriority prio = TaskPriority::NORMAL)
        : id(task_id), function_ptr(func), priority(prio), status(TaskStatus::PENDING),
          dependency_count(0), dep_index(0) {
        for (int i = 0; i < 8; ++i) {
            dependencies[i] = nullptr;
        }
    }
    
    /**
     * @brief Add a dependency to this task
     * @param dep Pointer to dependency task
     * @return true if dependency added successfully, false if array is full
     */
    bool add_dependency(Task* dep) {
        if (dep_index < 8) {
            dependencies[dep_index++] = dep;
            dependency_count++;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Check if all dependencies are completed
     * @return true if task is ready to execute
     */
    bool is_ready() const {
        if (dependency_count == 0) return true;
        
        for (uint8_t i = 0; i < dep_index; ++i) {
            if (dependencies[i] && dependencies[i]->status != TaskStatus::COMPLETED) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Scheduling strategy enumeration
 */
enum class SchedulingStrategy {
    FIFO,           ///< First In, First Out
    PRIORITY_BASED, ///< Priority-based scheduling
    ROUND_ROBIN     ///< Round-robin scheduling
};

/**
 * @brief Lightweight inference scheduler for task execution
 * 
 * This scheduler manages operator execution with support for:
 * - Task priorities and dependencies (simple DAG)
 * - Multiple scheduling strategies
 * - Thread-safe operations
 * - Fixed-size memory allocation
 */
class CMXScheduler {
private:
    static constexpr size_t MAX_TASKS = 256;    ///< Maximum number of tasks in queue
    static constexpr size_t MAX_READY_TASKS = 64; ///< Maximum ready tasks buffer
    
    Task task_pool_[MAX_TASKS];                 ///< Pre-allocated task pool
    Task* ready_queue_[MAX_READY_TASKS];        ///< Ready tasks queue
    size_t task_count_;                         ///< Current number of tasks
    size_t ready_count_;                        ///< Current number of ready tasks
    size_t ready_head_;                         ///< Ready queue head index
    size_t ready_tail_;                         ///< Ready queue tail index
    uint32_t next_task_id_;                     ///< Next available task ID
    
    SchedulingStrategy strategy_;               ///< Current scheduling strategy
    std::atomic<bool> is_running_;              ///< Scheduler running flag
    std::mutex queue_mutex_;                    ///< Mutex for thread safety
    
    /**
     * @brief Find next task to execute based on strategy
     * @return Pointer to next task, or nullptr if none available
     */
    Task* get_next_task();
    
    /**
     * @brief Update ready queue after task completion
     * @param completed_task Pointer to completed task
     */
    void update_ready_queue(Task* completed_task);
    
    /**
     * @brief Priority comparison for tasks
     * @param a First task
     * @param b Second task
     * @return true if task a has higher priority than task b
     */
    bool has_higher_priority(const Task* a, const Task* b) const;
    
    /**
     * @brief Add task to ready queue
     * @param task Pointer to task to add
     * @return true if successfully added, false if queue is full
     */
    bool enqueue_ready_task(Task* task);
    
    /**
     * @brief Remove task from ready queue
     * @return Pointer to dequeued task, or nullptr if queue is empty
     */
    Task* dequeue_ready_task();

public:
    /**
     * @brief Constructor
     * @param strategy Initial scheduling strategy
     */
    explicit CMXScheduler(SchedulingStrategy strategy = SchedulingStrategy::FIFO);
    
    /**
     * @brief Destructor
     */
    ~CMXScheduler();
    
    /**
     * @brief Initialize the scheduler
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown the scheduler
     */
    void shutdown();
    
    /**
     * @brief Submit a task for execution
     * @param func Function to execute
     * @param priority Task priority level
     * @return Task ID if successful, 0 if failed
     */
    uint32_t submit_task(std::function<void()> func, TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Submit a task with dependencies
     * @param func Function to execute
     * @param deps Array of dependency task IDs
     * @param dep_count Number of dependencies
     * @param priority Task priority level
     * @return Task ID if successful, 0 if failed
     */
    uint32_t submit_task_with_deps(std::function<void()> func, 
                                   const uint32_t* deps, 
                                   size_t dep_count,
                                   TaskPriority priority = TaskPriority::NORMAL);
    
    /**
     * @brief Execute all ready tasks
     * @return Number of tasks executed
     */
    size_t execute_ready_tasks();
    
    /**
     * @brief Execute a single task if available
     * @return true if a task was executed, false if none available
     */
    bool execute_single_task();
    
    /**
     * @brief Wait for all tasks to complete
     * @param timeout_ms Timeout in milliseconds (0 = no timeout)
     * @return true if all tasks completed, false if timeout occurred
     */
    bool wait_for_completion(uint32_t timeout_ms = 0);
    
    /**
     * @brief Get task by ID
     * @param task_id Task identifier
     * @return Pointer to task, or nullptr if not found
     */
    Task* get_task(uint32_t task_id);
    
    /**
     * @brief Check if scheduler has pending tasks
     * @return true if there are pending tasks
     */
    bool has_pending_tasks() const;
    
    /**
     * @brief Get number of pending tasks
     * @return Number of tasks in pending state
     */
    size_t get_pending_task_count() const;
    
    /**
     * @brief Get number of ready tasks
     * @return Number of tasks ready for execution
     */
    size_t get_ready_task_count() const;
    
    /**
     * @brief Set scheduling strategy
     * @param strategy New scheduling strategy
     */
    void set_strategy(SchedulingStrategy strategy);
    
    /**
     * @brief Get current scheduling strategy
     * @return Current scheduling strategy
     */
    SchedulingStrategy get_strategy() const { return strategy_; }
    
    /**
     * @brief Reset scheduler state
     */
    void reset();
    
    /**
     * @brief Get scheduler statistics
     * @param total_tasks Total number of tasks processed
     * @param completed_tasks Number of completed tasks
     * @param failed_tasks Number of failed tasks
     */
    void get_stats(size_t& total_tasks, size_t& completed_tasks, size_t& failed_tasks) const;
};

} // namespace runtime
} // namespace cmx