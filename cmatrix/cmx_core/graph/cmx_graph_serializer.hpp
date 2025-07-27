#pragma once

#include <cstdint>
#include <cstddef>

namespace cmx {
namespace graph {

// Forward declarations
class CMXGraph;

/**
 * @brief Serialization format version for compatibility checking
 */
enum class SerializationVersion : uint32_t {
    VERSION_1_0 = 0x01000000,
    CURRENT = VERSION_1_0
};

/**
 * @brief Serialization flags for controlling output format
 */
enum class SerializationFlags : uint32_t {
    NONE = 0,
    INCLUDE_DEBUG_INFO = 1 << 0,
    COMPRESS_WEIGHTS = 1 << 1,
    VALIDATE_CHECKSUMS = 1 << 2
};

/**
 * @brief Header structure for serialized graph files
 */
struct CMXSerializationHeader {
    uint32_t magic_number;      // 'CMXG' as uint32
    uint32_t version;           // SerializationVersion
    uint32_t flags;             // SerializationFlags
    uint32_t header_size;       // Size of this header
    uint32_t graph_size;        // Size of graph data
    uint32_t checksum;          // CRC32 of graph data
    uint32_t node_count;        // Number of nodes
    uint32_t tensor_count;      // Number of tensors
    uint32_t reserved[8];       // Reserved for future use
};

/**
 * @brief Result codes for serialization operations
 */
enum class SerializationResult {
    SUCCESS = 0,
    ERROR_INVALID_GRAPH,
    ERROR_BUFFER_TOO_SMALL,
    ERROR_INVALID_FORMAT,
    ERROR_CHECKSUM_MISMATCH,
    ERROR_UNSUPPORTED_VERSION,
    ERROR_IO_ERROR,
    ERROR_OUT_OF_MEMORY
};

/**
 * @brief Statistics about serialization operation
 */
struct SerializationStats {
    size_t original_size;
    size_t compressed_size;
    size_t nodes_serialized;
    size_t tensors_serialized;
    uint32_t checksum;
};

/**
 * @brief Graph serializer and deserializer for CMX runtime
 * 
 * Handles conversion of CMXGraph objects to/from binary format
 * with support for versioning, compression, and validation.
 */
class CMXGraphSerializer {
public:
    CMXGraphSerializer();
    ~CMXGraphSerializer();

    // Disable copy constructor and assignment
    CMXGraphSerializer(const CMXGraphSerializer&) = delete;
    CMXGraphSerializer& operator=(const CMXGraphSerializer&) = delete;

    /**
     * @brief Serialize graph to memory buffer
     * @param graph Graph to serialize
     * @param buffer Output buffer (must be pre-allocated)
     * @param buffer_size Size of output buffer
     * @param flags Serialization options
     * @param stats Optional statistics output
     * @return SerializationResult indicating success/failure
     */
    SerializationResult serialize_to_buffer(
        const CMXGraph& graph,
        uint8_t* buffer,
        size_t buffer_size,
        SerializationFlags flags = SerializationFlags::NONE,
        SerializationStats* stats = nullptr
    );

    /**
     * @brief Serialize graph to file
     * @param graph Graph to serialize
     * @param filename Output file path
     * @param flags Serialization options
     * @param stats Optional statistics output
     * @return SerializationResult indicating success/failure
     */
    SerializationResult serialize_to_file(
        const CMXGraph& graph,
        const char* filename,
        SerializationFlags flags = SerializationFlags::NONE,
        SerializationStats* stats = nullptr
    );

    /**
     * @brief Deserialize graph from memory buffer
     * @param buffer Input buffer containing serialized graph
     * @param buffer_size Size of input buffer
     * @param graph Output graph object
     * @param flags Deserialization options
     * @return SerializationResult indicating success/failure
     */
    SerializationResult deserialize_from_buffer(
        const uint8_t* buffer,
        size_t buffer_size,
        CMXGraph& graph,
        SerializationFlags flags = SerializationFlags::NONE
    );

    /**
     * @brief Deserialize graph from file
     * @param filename Input file path
     * @param graph Output graph object
     * @param flags Deserialization options
     * @return SerializationResult indicating success/failure
     */
    SerializationResult deserialize_from_file(
        const char* filename,
        CMXGraph& graph,
        SerializationFlags flags = SerializationFlags::NONE
    );

    /**
     * @brief Calculate required buffer size for serialization
     * @param graph Graph to serialize
     * @param flags Serialization options
     * @return Required buffer size in bytes, 0 on error
     */
    size_t calculate_serialized_size(
        const CMXGraph& graph,
        SerializationFlags flags = SerializationFlags::NONE
    );

    /**
     * @brief Validate serialized graph format
     * @param buffer Buffer containing serialized data
     * @param buffer_size Size of buffer
     * @return true if format is valid, false otherwise
     */
    bool validate_format(const uint8_t* buffer, size_t buffer_size);

    /**
     * @brief Get version information from serialized buffer
     * @param buffer Buffer containing serialized data
     * @param buffer_size Size of buffer
     * @return SerializationVersion or VERSION_1_0 on error
     */
    SerializationVersion get_version(const uint8_t* buffer, size_t buffer_size);

    /**
     * @brief Check if serialization version is supported
     * @param version Version to check
     * @return true if supported, false otherwise
     */
    static bool is_version_supported(SerializationVersion version);

    /**
     * @brief Get human-readable error message for result code
     * @param result Result code
     * @return Error message string
     */
    static const char* get_error_message(SerializationResult result);

private:
    /**
     * @brief Write graph header to buffer
     * @param buffer Output buffer
     * @param graph Graph being serialized
     * @param flags Serialization flags
     * @return Number of bytes written
     */
    size_t write_header(uint8_t* buffer, const CMXGraph& graph, SerializationFlags flags);

    /**
     * @brief Write graph nodes to buffer
     * @param buffer Output buffer
     * @param offset Current buffer offset
     * @param graph Graph being serialized
     * @return Number of bytes written
     */
    size_t write_nodes(uint8_t* buffer, size_t offset, const CMXGraph& graph);

    /**
     * @brief Write tensor metadata to buffer
     * @param buffer Output buffer
     * @param offset Current buffer offset
     * @param graph Graph being serialized
     * @return Number of bytes written
     */
    size_t write_tensors(uint8_t* buffer, size_t offset, const CMXGraph& graph);

    /**
     * @brief Write graph topology to buffer
     * @param buffer Output buffer
     * @param offset Current buffer offset
     * @param graph Graph being serialized
     * @return Number of bytes written
     */
    size_t write_topology(uint8_t* buffer, size_t offset, const CMXGraph& graph);

    /**
     * @brief Read and validate header from buffer
     * @param buffer Input buffer
     * @param buffer_size Buffer size
     * @param header Output header structure
     * @return SerializationResult indicating success/failure
     */
    SerializationResult read_header(
        const uint8_t* buffer,
        size_t buffer_size,
        CMXSerializationHeader& header
    );

    /**
     * @brief Read nodes from buffer
     * @param buffer Input buffer
     * @param offset Current buffer offset
     * @param graph Output graph
     * @param node_count Number of nodes to read
     * @return SerializationResult indicating success/failure
     */
    SerializationResult read_nodes(
        const uint8_t* buffer,
        size_t offset,
        CMXGraph& graph,
        uint32_t node_count
    );

    /**
     * @brief Read tensor metadata from buffer
     * @param buffer Input buffer
     * @param offset Current buffer offset
     * @param graph Output graph
     * @param tensor_count Number of tensors to read
     * @return SerializationResult indicating success/failure
     */
    SerializationResult read_tensors(
        const uint8_t* buffer,
        size_t offset,
        CMXGraph& graph,
        uint32_t tensor_count
    );

    /**
     * @brief Read graph topology from buffer
     * @param buffer Input buffer
     * @param offset Current buffer offset
     * @param graph Output graph
     * @return SerializationResult indicating success/failure
     */
    SerializationResult read_topology(
        const uint8_t* buffer,
        size_t offset,
        CMXGraph& graph
    );

    /**
     * @brief Calculate CRC32 checksum
     * @param data Data buffer
     * @param size Size of data
     * @return CRC32 checksum
     */
    uint32_t calculate_checksum(const uint8_t* data, size_t size);

    /**
     * @brief Compress data using simple RLE compression
     * @param input Input data
     * @param input_size Input size
     * @param output Output buffer
     * @param output_size Output buffer size
     * @return Compressed size, 0 on error
     */
    size_t compress_data(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_size
    );

    /**
     * @brief Decompress RLE compressed data
     * @param input Compressed data
     * @param input_size Compressed size
     * @param output Output buffer
     * @param output_size Output buffer size
     * @return Decompressed size, 0 on error
     */
    size_t decompress_data(
        const uint8_t* input,
        size_t input_size,
        uint8_t* output,
        size_t output_size
    );

    // Constants
    static constexpr uint32_t MAGIC_NUMBER = 0x474D5843; // 'CMXG'
    static constexpr size_t MAX_FILENAME_LENGTH = 256;
    static constexpr size_t BUFFER_ALIGNMENT = 8;

    // Internal state
    uint8_t* temp_buffer_;
    size_t temp_buffer_size_;
    bool initialized_;
};

/**
 * @brief Utility functions for serialization flags
 */
inline SerializationFlags operator|(SerializationFlags a, SerializationFlags b) {
    return static_cast<SerializationFlags>(
        static_cast<uint32_t>(a) | static_cast<uint32_t>(b)
    );
}

inline SerializationFlags operator&(SerializationFlags a, SerializationFlags b) {
    return static_cast<SerializationFlags>(
        static_cast<uint32_t>(a) & static_cast<uint32_t>(b)
    );
}

inline bool has_flag(SerializationFlags flags, SerializationFlags flag) {
    return (flags & flag) == flag;
}

} // namespace graph
} // namespace cmx