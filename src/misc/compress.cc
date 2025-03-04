
#include "compress.h"
#include "info.h"
#include "align.h"

extern "C"{

ncclResult_t launchCompress(const void* orgbuff, const size_t orgChunkCount, ncclDataType_t orgType,  void* compbuff, 
    const size_t compChunkCount, ncclDataType_t compType, const size_t numChunks, cudaStream_t stream);

ncclResult_t launchDecompress(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, void *decompbuff, 
    const size_t decompChunkCount, ncclDataType_t decompType, const size_t numChunks, cudaStream_t stream);

ncclResult_t launchReductionColl(const void* input1, const void* input2, void* output, ncclDataType_t type, size_t inputCount, 
    cudaStream_t stream);

ncclResult_t launchDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, void *output, 
    const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream);

ncclResult_t launchReduceChunk(const void* input, size_t chunkCount, void* output, ncclDataType_t type, int numChunks, 
    cudaStream_t stream);
}


ncclResult_t ncclCompress(const void* orgbuff, const size_t orgChunkCount, ncclDataType_t orgType,  void** compbuff, 
    size_t* compChunkCount, ncclDataType_t* compType, const size_t numChunks, cudaStream_t stream)
{
      // now only minmaxUint8
      *compType = ncclDataType_t::ncclUint8;
      if(orgType == ncclDataType_t::ncclFloat32){
        *compChunkCount = (alignUp(orgChunkCount, 32) + alignUp(4 * 2, 32));// 4 * 2 is max + min FP32
      }else if(orgType == ncclDataType_t::ncclFloat16){
        *compChunkCount = (alignUp(orgChunkCount, 32) + alignUp(2 * 2, 32));//FP16
      }
      
      if(*compbuff == nullptr)
        CUDACHECK(cudaMallocAsync((void**)compbuff, (*compChunkCount) * numChunks * ncclTypeSize(*compType), stream));
     

      NCCLCHECK(launchCompress(orgbuff, orgChunkCount, orgType, *compbuff, *compChunkCount, *compType, 
                              numChunks, stream));
      return ncclSuccess;
}


ncclResult_t ncclDecompress(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, void* decompbuff, 
    const size_t decompChunkCount, ncclDataType_t decompType, const size_t numChunks, cudaStream_t stream)
{
    NCCLCHECK(launchDecompress(compbuff, compChunkCount, compType, decompbuff, decompChunkCount, decompType, numChunks, stream));
    return ncclSuccess;
}

ncclResult_t ncclDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, void* output, 
    const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream)
{
    NCCLCHECK(launchDecompressReduce(compbuff, compChunkCount, compType, reduceInput, output, OutputChunkCount, OutputType, numChunks, stream));
    return ncclSuccess;
}
