
#include "nccl.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include "compress_kernel.h"
#include "reduce_extend_kernel.h"


const float eps = 1e-7;


template<typename T>
__device__ inline uint8_t __minmax_uint8_compress(T f, float scale, float lower_bound, float upper_bound) {
    float level = f * scale;
    level = min(level, upper_bound);
    return level - lower_bound;

}

template<>
__device__ inline uint8_t __minmax_uint8_compress<float>(float f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(f * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<>
__device__ inline uint8_t __minmax_uint8_compress<half>(half f, float scale, float lower_bound, float upper_bound) {
    float level = rintf(__half2float(f) * scale);
    level = min(level, upper_bound);
    return level - lower_bound;
}

template<typename T>
__device__ inline T __minmax_uint8_decompress(uint8_t i, float scale, float lower_bound, float upper_bound, T placeholder) {
    return (i + lower_bound) / scale;
}

template<>
__device__ inline half __minmax_uint8_decompress<half>(uint8_t i, float scale, float lower_bound, float upper_bound, half placeholder) {
    return __float2half((i + lower_bound) / scale);
}

template<typename T>
__device__ inline float __load_as_float(T * array) {
    return array[0];
}

template<>
__device__ inline float __load_as_float<half>(half * array) {
    return __half2float(array[0]);
}

template<typename T>
__device__ inline void __store_float(T * array, float data) {
    array[0] = data;
}

template<>
__device__ inline void __store_float<half>(half * array, float data) {
    array[0] = __float2half(data);
}


template<typename T>
__global__ void
compressFloattoUint8(const void *input, int chunkCount, int compChunkCount, void *output) {
    T* orgbuff= (T*)input;
    uint8_t* compbuff = (uint8_t*) output;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float min_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount));
    float max_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upperBound = rintf(max_ * scale);
    float lowerBound = upperBound - 255.0;
    for (int i = idx; i < chunkCount; i += blockDim.x * gridDim.x) {
        int k = idy * chunkCount + i;
        int o = idy * compChunkCount + 32 + i;
        compbuff[o] = __minmax_uint8_compress(orgbuff[k], scale, lowerBound, upperBound);
    }

    if (idx == 0) {
        // write max min to output buffer
        __store_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount), min_);
        __store_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount + sizeof(T)), max_);
    }
}

template<typename T>
__global__ void
decompressUint8toFloat(const void* input, int chunkCount, int compChunkCount, void *output) {
    uint8_t* compbuff = (uint8_t*) input;
    T* decompbuff = (T*) output;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float min_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount));
    const float max_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upperBound = rintf(max_ * scale);
    float lowerBound = upperBound - 255.0;

    for (int i = idx; i < chunkCount; i += blockDim.x * gridDim.x) {
        int k = idy * chunkCount + i;
        int o = idy * compChunkCount + 32 + i;
        decompbuff[k] = __minmax_uint8_decompress(compbuff[o], scale, lowerBound, upperBound, decompbuff[k]);
    }

}

template<typename T>
__global__ void
decompressUint8toFloatReduce(const void* compInput, const void* reduceInput, int chunkCount, int compChunkCount, void *output) {
    uint8_t* compbuff = (uint8_t*) compInput;
    T* reducebuff = (T*) reduceInput;
    T* outputbuff = (T*) output;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float min_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount));
    const float max_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * compChunkCount + sizeof(T)));

    float scale = 255.0 / (max_ - min_ + eps);
    float upperBound = rintf(max_ * scale);
    float lowerBound = upperBound - 255.0;

    for (int i = idx; i < chunkCount; i += blockDim.x * gridDim.x) {
        int k = idy * chunkCount + i;
        int o = idy * compChunkCount + 32 + i;
        outputbuff[k] = reducebuff[k] + __minmax_uint8_decompress(compbuff[o], scale, lowerBound, upperBound, outputbuff[k]);
    }

}

extern "C"{

ncclResult_t launchCompress(const void* orgbuff, const size_t orgChunkCount, ncclDataType_t orgType,  void* compbuff, 
    const size_t compChunkCount, ncclDataType_t compType, const size_t numChunks, cudaStream_t stream){
    int block = orgChunkCount < 1024 ? orgChunkCount : 1024;
    dim3 grid(DIVUP(orgChunkCount, block), numChunks);
    if(orgType == ncclDataType_t::ncclFloat16){
        // findMinMax<half>(orgbuff, orgChunkCount, compbuff, compChunkCount, numChunks, stream);
        // maxMinBlockReduce<half> <<<grid, block, 2 * block * sizeof(T), stream>>> (input, chunkCount, output, compChunkCount);
        compressFloattoUint8<half> <<<grid, block, 0, stream>>>(orgbuff, orgChunkCount, compChunkCount, compbuff);
    } else if(orgType == ncclDataType_t::ncclFloat32){
        
        minMaxReduction(orgbuff, orgChunkCount, compbuff, compChunkCount, numChunks, orgType, stream);
        compressFloattoUint8<float> <<<grid, block, 0, stream>>>(orgbuff, orgChunkCount, compChunkCount, compbuff);
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
}

ncclResult_t launchDecompress(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, void *decompbuff, 
    const size_t decompChunkCount, ncclDataType_t decompType, const size_t numChunks, cudaStream_t stream){

    int block = decompChunkCount < 1024 ? decompChunkCount : 1024;
    dim3 grid(DIVUP(decompChunkCount, block), numChunks);
    if(decompType == ncclDataType_t::ncclFloat16){
        decompressUint8toFloat<half> <<<grid, block, 0, stream>>>(compbuff, decompChunkCount, compChunkCount, decompbuff);
    } else if(decompType == ncclDataType_t::ncclFloat32){
        decompressUint8toFloat<float> <<<grid, block, 0, stream>>>(compbuff, decompChunkCount, compChunkCount, decompbuff);
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
}

ncclResult_t launchDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, void *output, 
    const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream){

    int block = OutputChunkCount < 1024 ? OutputChunkCount : 1024;
    dim3 grid(DIVUP(OutputChunkCount, block), numChunks);
    
    if(OutputType == ncclDataType_t::ncclFloat16){
        decompressUint8toFloatReduce<half> <<<grid, block, 0, stream>>>(compbuff, reduceInput, OutputChunkCount, compChunkCount, output);
    } else if(OutputType == ncclDataType_t::ncclFloat32){
        decompressUint8toFloatReduce<float> <<<grid, block, 0, stream>>>(compbuff, reduceInput, OutputChunkCount, compChunkCount, output);
    }
    CUDACHECK(cudaGetLastError());
    return ncclSuccess;
}


}