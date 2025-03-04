
#include "coll_extend_p2p.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"

NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAlltoAll(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

  NCCLCHECK(ncclGroupStart());
  for (size_t r = 0; r < comm->nRanks ; r++){
    char* r_sendbuf =(char*) sendbuff + r * sendcount*ncclTypeSize(datatype);
    char* r_recvbuf =(char*) recvbuff + r * sendcount*ncclTypeSize(datatype);
      // NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void *)r_recvbuf, sendcount, datatype, r, comm, stream));
      NCCLCHECK(ncclSend((void *)r_sendbuf, sendcount, datatype, r, comm, stream));
      // NCCLCHECK(ncclGroupEnd());
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterP2P, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterP2P(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
    
    int rightRank = (comm->rank + 1) % comm->nRanks;
    int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
    size_t chunkBytes = recvcount * ncclTypeSize(datatype);
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    void* recvTempbuff;
    void* sendTempbuff;
    CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * chunkBytes, stream));
    CUDACHECK(cudaMallocAsync((void**)&sendTempbuff, comm->nRanks * chunkBytes, stream));
    CUDACHECK(cudaMemcpyAsync(sendTempbuff, sendbuff, comm->nRanks * chunkBytes, cudaMemcpyDeviceToDevice, stream));
  
    for (int r = comm->nRanks - 1; r >= 0; r--) {
      int sendIdx = (comm->rank + r) % comm->nRanks;
      int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;
      
      char* r_sendbuf =(char*) sendTempbuff + sendIdx * chunkBytes;
      char* r_recvbuf =(char*) recvTempbuff + recvIdx * chunkBytes;
      if(r == comm->nRanks - 1){
        // Ring step 0
  
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, recvcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, recvcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
      } 
      else if(r > 0) {
        // Ring step 1 ~ N-2 
        // recv reduce send

        int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
        // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
        char* r_prevRecvbuf =(char*) recvTempbuff + prevRecvIdx * chunkBytes;
        NCCLCHECK(ncclReductionColl((void*)r_sendbuf, (void*)r_prevRecvbuf, 
              (void*)r_sendbuf, datatype, op, recvcount, stream));
        
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, recvcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, recvcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
      } 
      else {
        // Ring step N

        int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
        // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
        char* r_prevRecvbuf =(char*) recvTempbuff + prevRecvIdx * chunkBytes;
        NCCLCHECK(ncclReductionColl((void*)r_sendbuf, (void*)r_prevRecvbuf, 
              (void*)recvbuff, datatype, op, recvcount, stream));
      }
    }
  
    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
    CUDACHECK(cudaFreeAsync(sendTempbuff, stream));
  
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceRingP2P, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceRingP2P(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
   
  size_t chunkCount = count / comm->nRanks;
  char* r_recvbuf = (char*) recvbuff + comm->rank *  chunkCount * ncclTypeSize(datatype);
  NCCLCHECK(ncclReduceScatterP2P(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));
  NCCLCHECK(ncclAllGather(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));

  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclAllGatherRingP2P, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherRingP2P(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    int rightRank = (comm->rank + 1) % comm->nRanks;
    int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
    size_t chunkBytes = sendcount * ncclTypeSize(datatype);
    // step 0 local copy
    CUDACHECK(cudaMemcpyAsync((char*)recvbuff + comm->rank * chunkBytes, sendbuff, chunkBytes, cudaMemcpyDeviceToDevice, stream));
    // step 1 ~ N - 2  
    for(int r = 0; r < comm->nRanks - 1; r++){
        int sendIdx = (comm->rank - r + comm->nRanks) % comm->nRanks;
        int recvIdx = (comm->rank - (r + 1) + comm->nRanks) % comm->nRanks;
        char* r_sendbuf =(char*) recvbuff + sendIdx * chunkBytes;
        char* r_recvbuf =(char*) recvbuff + recvIdx * chunkBytes;
        NCCLCHECK(ncclGroupStart());
        NCCLCHECK(ncclRecv((void*)r_recvbuf, sendcount, datatype, leftRank, comm, stream));
        NCCLCHECK(ncclSend((void*)r_sendbuf, sendcount, datatype, rightRank, comm, stream));
        NCCLCHECK(ncclGroupEnd());
    }

    return ncclSuccess;
}
