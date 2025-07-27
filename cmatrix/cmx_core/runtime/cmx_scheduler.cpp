#include "cmx_scheduler.hpp"
#include <algorithm>
#include <chrono>
#include <thread>

namespace cmx {
namespace runtime {

CMXScheduler::CMXScheduler(SchedulingStrategy strategy)
    : task_count_(0), ready_count_(0), ready_head_(0), ready_tail_(0),
      next_task_id_(1), strategy_(strategy), is_running_(false) {
    // Initialize task pool
    for (size_t i = 0; i < MAX_TASKS; ++i) {
        task_pool_[i] = Task();
    }
    
    // Initialize ready queue
    for (size_t i = 0; i < MAX_READY_TASKS; ++i) {
        ready_queue_[i] = nullptr;
    }
}

CMXScheduler::~CMXScheduler() {
    shutdown();
}

bool CMXScheduler::initialize() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (is_running_.load()) {
        return false; // Already initialized
    }
    
    // Reset state
    task_count_ = 0;
    ready_count_ = 0;
    ready_head_ = 0;
    ready_tail_ = 0;
    next_task_id_ = 1;
    
    is_running_.store(true);
    return true;
}

void CMXScheduler::shutdown() {
    is_running_.store(false);
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Clear all tasks
    for (size_t i = 0; i < task_count_; ++i) {
        task_pool_[i] = Task();
    }
    
    // Clear ready queue
    for (size_t i = 0; i < MAX_READY_TASKS; ++i) {
        ready_queue_[i] = nullptr;
    }
    
    task_count_ = 0;
    ready_count_ = 0;
    ready_head_ = 0;
    ready_tail_ = 0;
}

uint32_t CMXScheduler::submit_task(std::function<void()> func, TaskPriority priority) {
    if (!is_running_.load() || !func) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (task_count_ >= MAX_TASKS) {
        return 0; // Task pool full
    }
    
    // Create new task
    uint32_t task_id = next_task_id_++;
    Task& task = task_pool_[task_count_++];
    task = Task(task_id, func, priority);
    
    // If task has no dependencies, add to ready queue
    if (task.is_ready()) {
        enqueue_ready_task(&task);
    }
    
    return task_id;
}

uint32_t CMXScheduler::submit_task_with_deps(std::function<void()> func,
                                            const uint32_t* deps,
                                            size_t dep_count,
                                            TaskPriority priority) {
    if (!is_running_.load() || !func || dep_count > 8) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (task_count_ >= MAX_TASKS) {
        return 0; // Task pool full
    }
    
    // Create new task
    uint32_t task_id = next_task_id_++;
    Task& task = task_pool_[task_count_++];
    task = Task(task_id, func, priority);
    
    // Add dependencies
    for (size_t i = 0; i < dep_count; ++i) {
        Task* dep_task = get_task(deps[i]);
        if (dep_task) {
            task.add_dependency(dep_task);
        }
    }
    
    // If task has no dependencies or all are completed, add to ready queue
    if (task.is_ready()) {
        enqueue_ready_task(&task);
    }
    
    return task_id;
}

bool CMXScheduler::execute_single_task() {
    if (!is_running_.load()) {
        return false;
    }
    
    Task* task = get_next_task();
    if (!task) {
        return false;
    }
    
    // Execute task
    task->status = TaskStatus::RUNNING;
    
    try {
        task->function_ptr();
        task->status = TaskStatus::COMPLETED;
    } catch (...) {
        task->status = TaskStatus::FAILED;
    }
    
    // Update ready queue with newly available tasks
    update_ready_queue(task);
    
    return true;
}

size_t CMXScheduler::execute_ready_tasks() {
    if (!is_running_.load()) {
        return 0;
    }
    
    size_t executed_count = 0;
    
    while (execute_single_task()) {
        ++executed_count;
    }
    
    return executed_count;
}

bool CMXScheduler::wait_for_completion(uint32_t timeout_ms) {
    if (!is_running_.load()) {
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    while (has_pending_tasks()) {
        // Execute available tasks
        execute_ready_tasks();
        
        // Check timeout
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time);
            if (elapsed.count() >= timeout_ms) {
                return false;
            }
        }
        
        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    
    return true;
}

Task* CMXScheduler::get_task(uint32_t task_id) {
    for (size_t i = 0; i < task_count_; ++i) {
        if (task_pool_[i].id == task_id) {
            return &task_pool_[i];
        }
    }
    return nullptr;
}

bool CMXScheduler::has_pending_tasks() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    for (size_t i = 0; i < task_count_; ++i) {
        if (task_pool_[i].status == TaskStatus::PENDING || 
            task_pool_[i].status == TaskStatus::RUNNING) {
            return true;
        }
    }
    
    return false;
}

size_t CMXScheduler::get_pending_task_count() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    size_t count = 0;
    for (size_t i = 0; i < task_count_; ++i) {
        if (task_pool_[i].status == TaskStatus::PENDING) {
            ++count;
        }
    }
    
    return count;
}

size_t CMXScheduler::get_ready_task_count() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return ready_count_;
}

void CMXScheduler::set_strategy(SchedulingStrategy strategy) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    strategy_ = strategy;
}

void CMXScheduler::reset() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Clear all tasks
    for (size_t i = 0; i < task_count_; ++i) {
        task_pool_[i] = Task();
    }
    
    // Clear ready queue
    for (size_t i = 0; i < MAX_READY_TASKS; ++i) {
        ready_queue_[i] = nullptr;
    }
    
    task_count_ = 0;
    ready_count_ = 0;
    ready_head_ = 0;
    ready_tail_ = 0;
    next_task_id_ = 1;
}

void CMXScheduler::get_stats(size_t& total_tasks, size_t& completed_tasks, size_t& failed_tasks) const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    total_tasks = task_count_;
    completed_tasks = 0;
    failed_tasks = 0;
    
    for (size_t i = 0; i < task_count_; ++i) {
        if (task_pool_[i].status == TaskStatus::COMPLETED) {
            ++completed_tasks;
        } else if (task_pool_[i].status == TaskStatus::FAILED) {
            ++failed_tasks;
        }
    }
}

Task* CMXScheduler::get_next_task() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (ready_count_ == 0) {
        return nullptr;
    }
    
    Task* selected_task = nullptr;
    
    switch (strategy_) {
        case SchedulingStrategy::FIFO:
            selected_task = dequeue_ready_task();
            break;
            
        case SchedulingStrategy::PRIORITY_BASED: {
            // Find highest priority task
            Task* highest_priority = nullptr;
            size_t highest_index = 0;
            
            for (size_t i = 0; i < ready_count_; ++i) {
                size_t index = (ready_head_ + i) % MAX_READY_TASKS;
                Task* task = ready_queue_[index];
                
                if (!highest_priority || has_higher_priority(task, highest_priority)) {
                    highest_priority = task;
                    highest_index = index;
                }
            }
            
            if (highest_priority) {
                // Remove from ready queue
                ready_queue_[highest_index] = nullptr;
                
                // Compact the queue
                for (size_t i = highest_index; i != ready_tail_; i = (i + 1) % MAX_READY_TASKS) {
                    size_t next = (i + 1) % MAX_READY_TASKS;
                    ready_queue_[i] = ready_queue_[next];
                }
                
                ready_tail_ = (ready_tail_ + MAX_READY_TASKS - 1) % MAX_READY_TASKS;
                ready_count_--;
                selected_task = highest_priority;
            }
            break;
        }
        
        case SchedulingStrategy::ROUND_ROBIN:
            // For simple round-robin, just use FIFO
            selected_task = dequeue_ready_task();
            break;
    }
    
    return selected_task;
}

void CMXScheduler::update_ready_queue(Task* completed_task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    // Check all tasks for newly available ones
    for (size_t i = 0; i < task_count_; ++i) {
        Task& task = task_pool_[i];
        
        if (task.status == TaskStatus::PENDING && task.is_ready()) {
            enqueue_ready_task(&task);
        }
    }
}

bool CMXScheduler::has_higher_priority(const Task* a, const Task* b) const {
    return static_cast<uint8_t>(a->priority) > static_cast<uint8_t>(b->priority);
}

bool CMXScheduler::enqueue_ready_task(Task* task) {
    if (ready_count_ >= MAX_READY_TASKS) {
        return false; // Queue full
    }
    
    ready_queue_[ready_tail_] = task;
    ready_tail_ = (ready_tail_ + 1) % MAX_READY_TASKS;
    ready_count_++;
    
    return true;
}

Task* CMXScheduler::dequeue_ready_task() {
    if (ready_count_ == 0) {
        return nullptr;
    }
    
    Task* task = ready_queue_[ready_head_];
    ready_queue_[ready_head_] = nullptr;
    ready_head_ = (ready_head_ + 1) % MAX_READY_TASKS;
    ready_count_--;
    
    return task;
}

} // namespace runtime
} // namespace cmx