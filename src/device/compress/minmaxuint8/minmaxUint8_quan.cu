
#include "nccl.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <limits>
#include "minmaxUint8_quan.h"

#define __hidden __attribute__ ((visibility("hidden")))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);


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
__device__ inline T __minmax_uint8_decompress(uint8_t i, float scale, float lower_bound, float upper_bound) {
    return (i + lower_bound) / scale;
}


template<>
__device__ inline half __minmax_uint8_decompress<half>(uint8_t i, float scale, float lower_bound, float upper_bound) {
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
__device__ inline T getInfinity();

template<>
__device__ inline float getInfinity<float>(){
    return INFINITY;
}

template<>
__device__ inline half getInfinity<half>(){
    return __float2half(INFINITY);
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
    __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    
    do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
    __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<typename T>
__device__ void warpReduce(volatile T* smin, volatile T* smax, const int tid, const int blockSize){
    if(blockSize >=64) {
        smin[tid] = smin[tid] < smin[tid + 32] ? smin[tid] : smin[tid + 32];
        smax[tid] = smax[tid] > smax[tid + 32] ? smax[tid] : smax[tid + 32];
    }
    if(blockSize >=32) {
        smin[tid] = smin[tid] < smin[tid + 16] ? smin[tid] : smin[tid + 16];
        smax[tid] = smax[tid] > smax[tid + 16] ? smax[tid] : smax[tid + 16];
    }
    if(blockSize >=16) {
        smin[tid] = smin[tid] < smin[tid + 8] ? smin[tid] : smin[tid + 8];
        smax[tid] = smax[tid] > smax[tid + 8] ? smax[tid] : smax[tid + 8];
    }
    if(blockSize >=8) {
        smin[tid] = smin[tid] < smin[tid + 4] ? smin[tid] : smin[tid + 4];
        smax[tid] = smax[tid] > smax[tid + 4] ? smax[tid] : smax[tid + 4];
    }
    if(blockSize >=4) {
        smin[tid] = smin[tid] < smin[tid + 2] ? smin[tid] : smin[tid + 2];
        smax[tid] = smax[tid] > smax[tid + 2] ? smax[tid] : smax[tid + 2];
    }
    if(blockSize >=2) {
        smin[tid] = smin[tid] < smin[tid + 1] ? smin[tid] : smin[tid + 1];
        smax[tid] = smax[tid] > smax[tid + 1] ? smax[tid] : smax[tid + 1];
    }
}

template<typename T>
__device__ void warpShuffleMinMax(T &localMin, T &localMax, unsigned int mask, int blockSize){
    if(blockSize >= 32){
        T tMax = __shfl_down_sync(mask, localMax, 16);
        T tMin = __shfl_down_sync(mask, localMin, 16);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    }
    if(blockSize >= 16){
        T tMin = __shfl_down_sync(mask, localMin, 8);
        T tMax = __shfl_down_sync(mask, localMax, 8);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    }
    if(blockSize >= 8){
        T tMin = __shfl_down_sync(mask, localMin, 4);
        T tMax = __shfl_down_sync(mask, localMax, 4);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
    if(blockSize >= 4){
        T tMin = __shfl_down_sync(mask, localMin, 2);
        T tMax = __shfl_down_sync(mask, localMax, 2);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
    if(blockSize >= 2){
        T tMin = __shfl_down_sync(mask, localMin, 1);
        T tMax = __shfl_down_sync(mask, localMax, 1);
        localMin = localMin < tMin ? localMin : tMin;
        localMax = localMax > tMax ? localMax : tMax;
    } 
}

template<typename T, size_t blockSize>
__global__ void 
maxMinBlockReduce(const void* input, const size_t chunkCount, void* output, const size_t compChunkCount){
    // extern __shared__  unsigned char smem[];
    __shared__ T smem[2 * blockSize];


    // T* sharedMem = reinterpret_cast<T*>(smem);
    T* sharedMem = smem;


    T* inputbuff = (T*)input;
    uint8_t* outputbuff = (uint8_t*)output;
    
    T* sharedMin = sharedMem;
    T* sharedMax = sharedMem + blockDim.x;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // const int blockSize = blockDim.x;


    T localMin = getInfinity<T>();
    T localMax = -getInfinity<T>();
    while(idx < chunkCount){
        int k = idy * chunkCount;
        localMax = (localMax > inputbuff[k + idx]) ? localMax : inputbuff[k + idx];
        localMin = (localMin < inputbuff[k + idx]) ? localMin : inputbuff[k + idx];
        idx += blockDim.x * gridDim.x;
    }

    sharedMax[tid] = localMax;
    sharedMin[tid] = localMin;

    __syncthreads();

    if(blockSize >= 1024){
        if(tid < 512){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 512]) ? localMax : sharedMax[tid + 512];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 512]) ? localMin : sharedMin[tid + 512];
        }
        __syncthreads();
    }

    if(blockSize >= 512){
        if(tid < 256){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 256]) ? localMax : sharedMax[tid + 256];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 256]) ? localMin : sharedMin[tid + 256];
        }
        __syncthreads();
    }

    if(blockSize >= 256){
        if(tid < 128){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 128]) ? localMax : sharedMax[tid + 128];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 128]) ? localMin : sharedMin[tid + 128];
        }
        __syncthreads();
    }

    if(blockSize >= 128){
        if(tid < 64){
            sharedMax[tid] = localMax = (localMax > sharedMax[tid + 64]) ? localMax : sharedMax[tid + 64];
            sharedMin[tid] = localMin = (localMin < sharedMin[tid + 64]) ? localMin : sharedMin[tid + 64];
        }
        __syncthreads();
    }

    // if(blockSize >= 64){
    //     if(tid < 32){
    //         sharedMax[tid] = localMax = (localMax > sharedMax[tid + 32]) ? localMax : sharedMax[tid + 32];
    //         sharedMin[tid] = localMin = (localMin < sharedMin[tid + 32]) ? localMin : sharedMin[tid + 32];
    //     }
    //     __syncthreads();
    // }
    // if(tid < 32){
    //     warpShuffleMinMax<T>(localMin, localMax, 0xffffffff, blockSize);
    // }

    if(tid < 32){
        warpReduce<T>(sharedMin, sharedMax, tid, blockSize);
    }

    if(tid == 0){

        // T localMax = sharedMax[0];
        // T localMin = sharedMin[0];
        // warpShuffleMinMax<T,blockDim.x>(localMin, localMax, 0xffffffff);
        T* min_ = reinterpret_cast<T *>(outputbuff + idy * compChunkCount);
        T* max_ = reinterpret_cast<T *>(outputbuff + idy * compChunkCount + sizeof(T));
        // atomicMin(min_, localMin);
        // atomicMax(max_, localMax);
        atomicMin(min_, sharedMin[0]);
        atomicMax(max_, sharedMax[0]);
    }
}

template <typename T>
__global__ void InitMinMax(void* input, const size_t chunkCount, const size_t numChunk){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint8_t* inputbuff = (uint8_t*) input;
    if(idx < numChunk){
        __store_float(reinterpret_cast<T *>(inputbuff + idx * chunkCount), getInfinity<T>());
        __store_float(reinterpret_cast<T *>(inputbuff + idx * chunkCount + sizeof(T)), -getInfinity<T>());
    }
}

template<typename T>
__global__ void
compressFloattoUint8(const void *input, int chunkCount, int compChunkCount, void *output) {
    // __shared__ T smem[2];

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
    // __shared__ T smem[2];
    uint8_t* compbuff = (uint8_t*) input;
    T* decompbuff = (T*) output;
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
        decompbuff[k] = __minmax_uint8_decompress<T>(compbuff[o], scale, lowerBound, upperBound);
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
        outputbuff[k] = reducebuff[k] + __minmax_uint8_decompress<T>(compbuff[o], scale, lowerBound, upperBound);
    }

}

struct minmaxUint8Config{
    /* custom configs*/
    int groupCount=2048;
} ;

__hidden void parseMinmaxUint8Config(const char* configFile, void** compConfig, int nodes, int devicesPerNodes){
    // alloc memory for config
    *compConfig = (void*) malloc(sizeof(minmaxUint8Config));
    minmaxUint8Config* config = reinterpret_cast<minmaxUint8Config*>(*compConfig);
    // default values
    config->groupCount = 2048;
    if(!configFile) return;
    // load config from file
    std::pair<const char*, const char*>* configPairs = nullptr;
    int configPairCount = 0;
    loadConfigPair(configFile, &configPairs, &configPairCount); 
    if(configPairs == nullptr) return;
    // get configs
    for(int i = 0; i < configPairCount; i++){
        // groupCounts
        if(strcmp(configPairs[i].first, "groupCount") == 0){
            char* end;
            long groupCount = strtol(configPairs[i].second, &end, 10);
            if(*end == '\0'){
            config->groupCount = static_cast<int>(groupCount);
            }
        }
    }
}


__hidden cudaError_t launchCompress(const void* orgbuff, void** compbuff, const size_t orgChunkCount, ncclDataType_t orgDayatype, 
                                    size_t* compChunkCount, ncclDataType_t* compDatatype, const size_t numChunks,
                                    void* config, cudaMemPool_t compMemPool, cudaStream_t stream)
{

    *compDatatype = ncclDataType_t::ncclUint8;
    if(orgDayatype == ncclDataType_t::ncclFloat32){
        size_t minmaxBytes = 4 * 2;
        ALIGN_SIZE(minmaxBytes, 32);
        size_t quanBytes = orgChunkCount;
        ALIGN_SIZE(quanBytes, 32);

        *compChunkCount = minmaxBytes + quanBytes; // 4 * 2 is max + min FP32
    } else if(orgDayatype == ncclDataType_t::ncclFloat16){
        size_t minmaxBytes = 2 * 2;
        ALIGN_SIZE(minmaxBytes, 32);
        size_t quanBytes = orgChunkCount;
        ALIGN_SIZE(quanBytes, 32);
        *compChunkCount = minmaxBytes + quanBytes; //FP16
    }
    if(*compbuff == nullptr || *compbuff == NULL){
        if(compMemPool != NULL)
            cudaMallocFromPoolAsync((void**)compbuff, (*compChunkCount) * numChunks, compMemPool, stream);
        else 
            cudaMallocAsync((void**)compbuff, (*compChunkCount) * numChunks, stream);
    }

    int InitBlock = numChunks < 1024 ? numChunks: 1024;
    int InitGrid = DIVUP(numChunks, InitBlock);

    int block = orgChunkCount < 1024 ? orgChunkCount : 1024;
    dim3 grid(128, numChunks);

    if(orgDayatype == ncclDataType_t::ncclFloat16){
        // findMinMax<half>(orgbuff, orgChunkCount, compbuff, compChunkCount, numChunks, stream);
        // maxMinBlockReduce<half> <<<grid, block, 2 * block * sizeof(T), stream>>> (input, chunkCount, output, compChunkCount);
        compressFloattoUint8<half> <<<grid, block, 0, stream>>>(orgbuff, orgChunkCount, *compChunkCount, *compbuff);
    } else if(orgDayatype == ncclDataType_t::ncclFloat32){
        // minMaxReduction(orgbuff, orgChunkCount, compbuff, compChunkCount, numChunks, orgDayatype, stream);
        InitMinMax<float> <<<InitGrid, InitBlock, 0, stream>>> (*compbuff, *compChunkCount, numChunks);
        maxMinBlockReduce<float, 1024> <<<grid, block, 0, stream>>> (orgbuff, orgChunkCount, *compbuff, *compChunkCount);
        compressFloattoUint8<float> <<<grid, block, 0, stream>>>(orgbuff, orgChunkCount, *compChunkCount, *compbuff);
    }
    return cudaGetLastError();
}

__hidden cudaError_t launchDecompress(void* decompbuff, const void* compbuff, const size_t decompChunkCount, ncclDataType_t decompDatatype, 
                                    const size_t compChunkCount, ncclDataType_t compDatatype, const size_t numChunks, void* config, 
                                    cudaStream_t stream)
{
    int block = decompChunkCount < 1024 ? decompChunkCount : 1024;
    // dim3 grid(DIVUP(decompChunkCount, 32 * block), numChunks);

    // dim3 grid(32, numChunks);
    dim3 grid(128, numChunks);

    if(decompDatatype == ncclDataType_t::ncclFloat16){
        decompressUint8toFloat<half> <<<grid, block, 0, stream>>>(compbuff, decompChunkCount, compChunkCount, decompbuff);
    } else if(decompDatatype == ncclDataType_t::ncclFloat32){
        decompressUint8toFloat<float> <<<grid, block, 0, stream>>>(compbuff, decompChunkCount, compChunkCount, decompbuff);
    }
    return cudaGetLastError();
}

__hidden cudaError_t launchDecompReduce(void* reducebuff, const void* compbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
    const size_t reduceChunkCount, ncclDataType_t reduceDataType, const size_t numChunks, void* config,
    cudaStream_t stream)
{
    return cudaSuccess;
}


__hidden cudaError_t launchDequanReduceQuan(const void* compbuff, void** recompbuff, const size_t compChunkCount, ncclDataType_t compDatatype,
    size_t* reCompChunkCount, ncclDataType_t* reCompDatatype, const size_t numChunks, void* config,
    cudaMemPool_t compMemPool, cudaStream_t stream)
{
    return cudaSuccess;

}


extern "C" const ncclCompressor_t minmaxUint8{
    .name = "minmaxUint8",
    .compress = launchCompress,
    .decompress = launchDecompress,
    .decompReduce = launchDecompReduce,
    .decompReduceComp = launchDequanReduceQuan,
    .parseConfig = parseMinmaxUint8Config
};

// extern "C"{

// ncclResult_t launchDecompressReduce(const void* compbuff, const size_t compChunkCount, ncclDataType_t compType, const void* reduceInput, void *output, 
//     const size_t OutputChunkCount, ncclDataType_t OutputType, const size_t numChunks, cudaStream_t stream){

//     int block = OutputChunkCount < 1024 ? OutputChunkCount : 1024;
//     dim3 grid(DIVUP(OutputChunkCount, block), numChunks);
    
//     if(OutputType == ncclDataType_t::ncclFloat16){
//         decompressUint8toFloatReduce<half> <<<grid, block, 0, stream>>>(compbuff, reduceInput, OutputChunkCount, compChunkCount, output);
//     } else if(OutputType == ncclDataType_t::ncclFloat32){
//         decompressUint8toFloatReduce<float> <<<grid, block, 0, stream>>>(compbuff, reduceInput, OutputChunkCount, compChunkCount, output);
//     }
//     CUDACHECK(cudaGetLastError());
//     return ncclSuccess;
// }


// }