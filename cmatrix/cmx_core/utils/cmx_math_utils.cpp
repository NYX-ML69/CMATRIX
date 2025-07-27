#include "cmx_math_utils.hpp"

namespace cmx {
namespace utils {

float clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

float min(float a, float b) {
    return (a < b) ? a : b;
}

float max(float a, float b) {
    return (a > b) ? a : b;
}

float abs(float value) {
    return (value < 0.0f) ? -value : value;
}

float fast_exp(float x) {
    // Fast exp approximation using polynomial
    // Clamp input to reasonable range
    x = clamp(x, -10.0f, 10.0f);
    
    // Use identity: exp(x) = exp(floor(x)) * exp(x - floor(x))
    int i = static_cast<int>(x);
    float f = x - i;
    
    // Approximate exp(f) for f in [0,1] using polynomial
    float exp_f = 1.0f + f * (1.0f + f * (0.5f + f * (0.1666f + f * 0.0417f)));
    
    // Multiply by exp(i) using bit manipulation for powers of 2
    if (i > 0) {
        for (int j = 0; j < i; ++j) {
            exp_f *= 2.718281f;
        }
    } else if (i < 0) {
        for (int j = 0; j < -i; ++j) {
            exp_f *= 0.3678794f;
        }
    }
    
    return exp_f;
}

float fast_log(float x) {
    if (x <= 0.0f) return -10.0f; // Return large negative for invalid input
    
    // Use bit manipulation to extract exponent
    union { float f; unsigned int i; } u;
    u.f = x;
    
    int exp = ((u.i >> 23) & 0xFF) - 127;
    u.i = (u.i & 0x007FFFFF) | 0x3F800000; // Normalize mantissa
    
    float y = u.f;
    
    // Polynomial approximation for log(1+x) where x is near 0
    float log_mantissa = (y - 1.0f) * (2.0f - 0.6666f * (y - 1.0f));
    
    return 0.693147f * exp + log_mantissa;
}

float fast_tanh(float x) {
    // Rational approximation
    if (x > 3.0f) return 1.0f;
    if (x < -3.0f) return -1.0f;
    
    float x2 = x * x;
    float numerator = x * (27.0f + x2);
    float denominator = 27.0f + 9.0f * x2;
    
    return numerator / denominator;
}

float fast_sigmoid(float x) {
    // Use fast_tanh: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    return 0.5f * (1.0f + fast_tanh(x * 0.5f));
}

float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    
    // Newton-Raphson iteration starting with bit manipulation estimate
    union { float f; unsigned int i; } u;
    u.f = x;
    u.i = (u.i >> 1) + 0x20000000; // Initial guess
    
    float y = u.f;
    
    // Two iterations of Newton-Raphson
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    
    return y;
}

float fast_rsqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    
    // Famous "fast inverse square root" algorithm
    union { float f; unsigned int i; } u;
    u.f = x;
    u.i = 0x5f3759df - (u.i >> 1);
    
    float y = u.f;
    
    // One iteration of Newton-Raphson
    y = y * (1.5f - 0.5f * x * y * y);
    
    return y;
}

float fast_pow(float base, float exp) {
    if (base <= 0.0f) return 0.0f;
    
    // Use identity: a^b = exp(b * ln(a))
    return fast_exp(exp * fast_log(base));
}

float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float leaky_relu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

float elu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * (fast_exp(x) - 1.0f);
}

float gelu(float x) {
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = 0.7978845f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + fast_tanh(inner));
}

float swish(float x) {
    return x * fast_sigmoid(x);
}

float softplus(float x) {
    // Avoid overflow for large x
    if (x > 20.0f) return x;
    return fast_log(1.0f + fast_exp(x));
}

void softmax(const float* input, float* output, int size) {
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = fast_exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

void layer_norm(const float* input, float* output, int size, float epsilon) {
    // Calculate mean
    float mean_val = mean(input, size);
    
    // Calculate variance
    float var = variance(input, size, mean_val);
    
    // Normalize
    float inv_std = fast_rsqrt(var + epsilon);
    for (int i = 0; i < size; ++i) {
        output[i] = (input[i] - mean_val) * inv_std;
    }
}

float mean(const float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum / size;
}

float variance(const float* input, int size, float mean_val) {
    if (mean_val == 0.0f) {
        mean_val = mean(input, size);
    }
    
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i] - mean_val;
        sum_sq_diff += diff * diff;
    }
    
    return sum_sq_diff / size;
}

float std_dev(const float* input, int size, float mean_val) {
    float var = variance(input, size, mean_val);
    return fast_sqrt(var);
}

void add(const float* a, const float* b, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}

void subtract(const float* a, const float* b, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = a[i] - b[i];
    }
}

void multiply(const float* a, const float* b, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = a[i] * b[i];
    }
}

void divide(const float* a, const float* b, float* output, int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = (b[i] != 0.0f) ? a[i] / b[i] : 0.0f;
    }
}

void scale(const float* input, float* output, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

void add_bias(const float* input, float* output, int size, float bias) {
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] + bias;
    }
}

} // namespace utils
} // namespace cmx