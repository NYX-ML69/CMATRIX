#include "cmx_api.hpp"

/**
 * @file cmx_api.cpp
 * @brief Implementation of core API utility functions
 */

namespace cmx {

const char* cmx_status_to_string(cmx_status status) {
    switch (status) {
        case cmx_status::OK:
            return "OK";
        case cmx_status::ERROR:
            return "ERROR";
        case cmx_status::INVALID_MODEL:
            return "INVALID_MODEL";
        case cmx_status::INVALID_HANDLE:
            return "INVALID_HANDLE";
        case cmx_status::MEMORY_ERROR:
            return "MEMORY_ERROR";
        case cmx_status::IO_ERROR:
            return "IO_ERROR";
        case cmx_status::NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case cmx_status::ALREADY_INITIALIZED:
            return "ALREADY_INITIALIZED";
        case cmx_status::UNSUPPORTED_VERSION:
            return "UNSUPPORTED_VERSION";
        case cmx_status::RUNTIME_ERROR:
            return "RUNTIME_ERROR";
        default:
            return "UNKNOWN_STATUS";
    }
}

} // namespace cmx