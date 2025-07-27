#include "cmx_pooling.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace cmx {
namespace kernels {

CmxPooling::CmxPooling()
    : pool_type_(PoolingType::MAX_POOL)
    , window_height_(0)
    , window_width_(0)
    , stride_height_(0)
    , stride_width_(0)
    , padding_(PaddingMode::VALID)
    , input_height_(0)
    , input_width_(0)
    , input_channels_(0)
    , batch_size_(0)
    , output_height_(0)
    , output_width_(0)
    , pad_top_(0)
    , pad_bottom_(0)
    , pad_left_(0)
    , pad_right_(0)
    , configured_(false) {
}

CmxPooling::~CmxPooling() {
}

bool CmxPooling::configure(PoolingType pool_type,
                          uint32_t window_height, uint32_t window_width,
                          uint32_t stride_height, uint32_t stride_width,
                          PaddingMode padding,
                          uint32_t input_height, uint32_t input_width,
                          uint32_t input_channels, uint32_t batch_size) {
    // Validate input parameters
    if (window_height == 0 || window_width == 0 ||
        stride_height == 0 || stride_width == 0 ||
        input_height == 0 || input_width == 0 ||
        input_channels == 0 || batch_size == 0) {
        return false;
    }
    
    // Store configuration
    pool_type_ = pool_type;
    window_height_ = window_height;
    window_width_ = window_width;
    stride_height_ = stride_height;
    stride_width_ = stride_width;
    padding_ = padding;
    input_height_ = input_height;
    input_width_ = input_width;
    input_channels_ = input_channels;
    batch_size_ = batch_size;
    
    // Calculate padding and output dimensions
    calculate_padding();
    
    // Calculate output dimensions
    output_height_ = (input_height_ + pad_top_ + pad_bottom_ - window_height_) / stride_height_ + 1;
    output_width_ = (input_width_ + pad_left_ + pad_right_ - window_width_) / stride_width_ + 1;
    
    configured_ = true;
    return true;
}

bool CmxPooling::run(const float* input_data, float* output_data) {
    if (!configured_ || !input_data || !output_data) {
        return false;
    }
    
    switch (pool_type_) {
        case PoolingType::MAX_POOL:
            execute_max_pool(input_data, output_data);
            break;
        case PoolingType::AVG_POOL:
            execute_avg_pool(input_data, output_data);
            break;
        default:
            return false;
    }
    
    return true;
}

bool CmxPooling::infer_shape(uint32_t& output_height, uint32_t& output_width,
                            uint32_t& output_channels, uint32_t& output_batch) const {
    if (!configured_) {
        return false;
    }
    
    output_height = output_height_;
    output_width = output_width_;
    output_channels = input_channels_;
    output_batch = batch_size_;
    
    return true;
}

size_t CmxPooling::get_memory_requirements() const {
    if (!configured_) {
        return 0;
    }
    
    // No additional memory required for pooling operations
    return 0;
}

void CmxPooling::calculate_padding() {
    if (padding_ == PaddingMode::VALID) {
        pad_top_ = pad_bottom_ = pad_left_ = pad_right_ = 0;
    } else { // SAME padding
        uint32_t pad_along_height = 0;
        uint32_t pad_along_width = 0;
        
        if (input_height_ % stride_height_ == 0) {
            pad_along_height = std::max(0U, window_height_ - stride_height_);
        } else {
            pad_along_height = std::max(0U, window_height_ - (input_height_ % stride_height_));
        }
        
        if (input_width_ % stride_width_ == 0) {
            pad_along_width = std::max(0U, window_width_ - stride_width_);
        } else {
            pad_along_width = std::max(0U, window_width_ - (input_width_ % stride_width_));
        }
        
        pad_top_ = pad_along_height / 2;
        pad_bottom_ = pad_along_height - pad_top_;
        pad_left_ = pad_along_width / 2;
        pad_right_ = pad_along_width - pad_left_;
    }
}

void CmxPooling::execute_max_pool(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t out_h = 0; out_h < output_height_; ++out_h) {
            for (uint32_t out_w = 0; out_w < output_width_; ++out_w) {
                for (uint32_t c = 0; c < input_channels_; ++c) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    int32_t h_start = static_cast<int32_t>(out_h * stride_height_) - static_cast<int32_t>(pad_top_);
                    int32_t w_start = static_cast<int32_t>(out_w * stride_width_) - static_cast<int32_t>(pad_left_);
                    
                    for (uint32_t kh = 0; kh < window_height_; ++kh) {
                        for (uint32_t kw = 0; kw < window_width_; ++kw) {
                            int32_t h_idx = h_start + static_cast<int32_t>(kh);
                            int32_t w_idx = w_start + static_cast<int32_t>(kw);
                            
                            float val = get_input_value(input_data, batch, h_idx, w_idx, c);
                            max_val = std::max(max_val, val);
                        }
                    }
                    
                    uint32_t output_idx = ((batch * output_height_ + out_h) * output_width_ + out_w) * input_channels_ + c;
                    output_data[output_idx] = max_val;
                }
            }
        }
    }
}

void CmxPooling::execute_avg_pool(const float* input_data, float* output_data) {
    for (uint32_t batch = 0; batch < batch_size_; ++batch) {
        for (uint32_t out_h = 0; out_h < output_height_; ++out_h) {
            for (uint32_t out_w = 0; out_w < output_width_; ++out_w) {
                for (uint32_t c = 0; c < input_channels_; ++c) {
                    float sum = 0.0f;
                    uint32_t count = 0;
                    
                    int32_t h_start = static_cast<int32_t>(out_h * stride_height_) - static_cast<int32_t>(pad_top_);
                    int32_t w_start = static_cast<int32_t>(out_w * stride_width_) - static_cast<int32_t>(pad_left_);
                    
                    for (uint32_t kh = 0; kh < window_height_; ++kh) {
                        for (uint32_t kw = 0; kw < window_width_; ++kw) {
                            int32_t h_idx = h_start + static_cast<int32_t>(kh);
                            int32_t w_idx = w_start + static_cast<int32_t>(kw);
                            
                            if (h_idx >= 0 && h_idx < static_cast<int32_t>(input_height_) &&
                                w_idx >= 0 && w_idx < static_cast<int32_t>(input_width_)) {
                                float val = get_input_value(input_data, batch, h_idx, w_idx, c);
                                sum += val;
                                count++;
                            }
                        }
                    }
                    
                    uint32_t output_idx = ((batch * output_height_ + out_h) * output_width_ + out_w) * input_channels_ + c;
                    output_data[output_idx] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
                }
            }
        }
    }
}

float CmxPooling::get_input_value(const float* input_data, uint32_t batch,
                                 int32_t h, int32_t w, uint32_t c) const {
    if (h < 0 || h >= static_cast<int32_t>(input_height_) ||
        w < 0 || w >= static_cast<int32_t>(input_width_)) {
        return 0.0f; // Padding value
    }
    
    uint32_t input_idx = ((batch * input_height_ + static_cast<uint32_t>(h)) * input_width_ + static_cast<uint32_t>(w)) * input_channels_ + c;
    return input_data[input_idx];
}

} // namespace kernels
} // namespace cmx