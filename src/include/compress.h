#ifndef NCCL_COMPRESS_H_
#define NCCL_COMPRESS_H_

#include "device.h"
#include "core.h"

ncclResult_t ncclCompress(const void* orgbuff, const size_t orgChunkCount, ncclDataType_t orgType,  void** compbuff, 
    size_t* compChunkCount, ncclDataType_t* compType, const size_t numChunks, cudaStream_t stream);

ncclResult_t ncclDecompress(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, void* decompbuff, 
    const size_t decompChunkCount, ncclDataType_t decompType, const size_t numChunks, cudaStream_t stream);

ncclResult_t ncclDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, 
    void* output, const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream);

#endif