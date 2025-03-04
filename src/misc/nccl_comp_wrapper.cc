
#include "nccl_comp_wrapper.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"

NCCL_API(ncclResult_t, ncclAllGatherComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  void* sendCompbuff=nullptr;
  void* recvCompbuff=nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  NCCLCHECK(ncclCompress(sendbuff, sendcount, datatype, &sendCompbuff, &compSendCount, &compDatatype, 1, stream));
  CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));

  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendCompbuff, recvCompbuff, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));

  // Free
  CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGatherCompRing, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherCompRing(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

  // Compress
  void* sendCompbuff=nullptr;
  void* recvCompbuff=nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  NCCLCHECK(ncclCompress(sendbuff, sendcount, datatype, &sendCompbuff, &compSendCount, &compDatatype, 1, stream));
  CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));

  // p2p base allgather
  int rightRank = (comm->rank + 1) % comm->nRanks;
  int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  size_t compChunkBytes = compSendCount * ncclTypeSize(compDatatype);
  // step 0 local copy
  CUDACHECK(cudaMemcpyAsync((char*)recvCompbuff + comm->rank * compChunkBytes, sendCompbuff, compChunkBytes, cudaMemcpyDeviceToDevice, stream));
  // CUDACHECK(cudaMemcpyAsync((char*)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype), sendbuff, sendcount * ncclTypeSize(datatype), cudaMemcpyDeviceToDevice, stream));

  // step 1 ~ N - 2  
  for(int r = 0; r < comm->nRanks - 1; r++){
      int sendIdx = (comm->rank - r + comm->nRanks) % comm->nRanks;
      int recvIdx = (comm->rank - (r + 1) + comm->nRanks) % comm->nRanks;
      char* r_sendbuf =(char*) recvCompbuff + sendIdx * compChunkBytes;
      char* r_recvbuf =(char*) recvCompbuff + recvIdx * compChunkBytes;
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)r_recvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)r_sendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
      // // Decompress
      // char* r_decompbuf =(char*) recvbuff + recvIdx * sendcount * ncclTypeSize(datatype);
      // NCCLCHECK(ncclDecompress((void*)r_recvbuf, compSendCount, compDatatype, r_decompbuf, sendcount, datatype, 1, stream));
  }

  // Decompress
  NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));

  // Free
  CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff, stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAlltoAllComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAlltoAllComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  void* sendCompbuff = nullptr;
  void* recvCompbuff = nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  NCCLCHECK(ncclCompress(sendbuff, sendcount, datatype, &sendCompbuff, &compSendCount, &compDatatype, comm->nRanks, stream));
  CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));

  NCCLCHECK(ncclAlltoAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  // Decompress
  NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));

  // Free
  CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff,stream));
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterComp, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterComp(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  int rightRank = (comm->rank + 1) % comm->nRanks;
  int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  size_t chunkBytes = recvcount * ncclTypeSize(datatype);
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  void *sendCompbuff;
  void *recvCompbuff;
  void *reducebuff;
  // Pre compress info
  size_t compSendCount = (alignUp(recvcount, 32) + alignUp(4 * 2, 32));
  ncclDataType_t compDatatype=ncclDataType_t::ncclUint8;

  CUDACHECK(cudaMallocAsync((void**)&sendCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  CUDACHECK(cudaMallocAsync((void**)&reducebuff, recvcount * ncclTypeSize(datatype), stream));

  for (int r = comm->nRanks - 1; r >= 0; r--) {
    // Ring step 0
    // compress - recv -  send

    int sendIdx = (comm->rank + r) % comm->nRanks;
    int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;
    char* r_sendbuf =(char*) sendCompbuff + sendIdx * compSendCount * ncclTypeSize(compDatatype);
    char* r_recvbuf =(char*) recvCompbuff + recvIdx * compSendCount * ncclTypeSize(compDatatype);
    if(r == comm->nRanks - 1){
      NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * chunkBytes, recvcount, datatype, 
                (void**)&r_sendbuf, &compSendCount, &compDatatype, 1, stream));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)r_recvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)r_sendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());

    } else if(r > 0) {
      // Ring step 1 ~ N - 2
      // recv - decompress - reduce - compress - send

      int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
      // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
      char* r_prevRecvbuf =(char*) recvCompbuff + prevRecvIdx * compSendCount * ncclTypeSize(compDatatype);
      NCCLCHECK(ncclDecompressReduce((void*)r_prevRecvbuf, compSendCount, compDatatype, (char*) sendbuff + sendIdx * chunkBytes, 
                (void*)reducebuff, recvcount, datatype, 1, stream));
      NCCLCHECK(ncclCompress((void*)reducebuff, recvcount, datatype, (void**)&r_sendbuf, &compSendCount, &compDatatype, 1, stream));
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)r_recvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)r_sendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    } else {
      // Ring step N - 1
      // decompress - reduce

      int prevRecvIdx = (recvIdx + 1) % comm->nRanks;
      // int nextRecvIdx = (recvIdx - 1 + comm->nRanks) % comm->nRanks;
      char* r_prevRecvbuf =(char*) recvCompbuff + prevRecvIdx * compSendCount * ncclTypeSize(compDatatype);
      NCCLCHECK(ncclDecompressReduce((void*)r_prevRecvbuf, compSendCount, compDatatype, (char*)sendbuff+sendIdx * chunkBytes, 
                (void*)recvbuff, recvcount, datatype, 1, stream));
    }
  }
  CUDACHECK(cudaFreeAsync(reducebuff, stream));
  CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompRing, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompRing(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  size_t chunkCount = count / comm->nRanks;
  char* r_recvbuf = (char*) recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);
  // reduce scatter comp
  NCCLCHECK(ncclReduceScatterComp(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));
  // allgather comp
  NCCLCHECK(ncclAllGatherComp(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));

  return ncclSuccess;
}
