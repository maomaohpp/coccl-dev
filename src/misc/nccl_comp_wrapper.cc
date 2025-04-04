
#include "nccl_comp_wrapper.h"
#include "nccl.h"
#include "argcheck.h"
#include "enqueue.h"
#include "compress.h"
#include "reduce_extend.h"
#include "compressor.h"

__thread struct parComm* parcomms = nullptr;
extern cudaMemPool_t* compMemPool;
extern size_t compMemPoolCnt;
__thread void* compBuff = nullptr;
__thread void* compBuffHandle = nullptr;
__thread size_t maxCompBuffBytes = 0;
// maxSendSize for allgather
__thread size_t aGMaxSendBytes = 0;

static ncclResult_t allocAndRegCompBuff(ncclComm_t comm, size_t bufferBytes){
    if(compBuffHandle == nullptr || bufferBytes > maxCompBuffBytes){
      CUDACHECK(cudaDeviceSynchronize());
      if(compBuffHandle!=nullptr && bufferBytes > maxCompBuffBytes){
          NCCLCHECK(ncclCommDeregister(comm, compBuffHandle));
          NCCLCHECK(ncclMemFree(compBuff));
          compBuff = nullptr;
          compBuffHandle = nullptr;
      }
      maxCompBuffBytes = bufferBytes;
      NCCLCHECK(ncclMemAlloc(&compBuff, bufferBytes));
      NCCLCHECK(ncclCommRegister(comm, compBuff, bufferBytes, &compBuffHandle));
      CUDACHECK(cudaDeviceSynchronize());
    }
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllGatherComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  // void* sendCompbuff=nullptr;
  // void* recvCompbuff=nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  
  // NCCLCHECK(ncclCompress(sendbuff, &recvbuff, sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  size_t totalSendBytes = comm->nRanks * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aGMaxSendBytes;

  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff: &compBuff, 
            sendcount, datatype, &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  // update the hold comp buffer
  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes = compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    NCCLCHECK(allocAndRegCompBuff(comm, compBuffBytes));
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  // INFO(NCCL_INIT, "coccl Allgather comp");
  // Gather
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    compBuff, compBuff, compSendCount, compDatatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, compBuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));


  // Free
  // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  return ncclSuccess;
}

// TODO inter- and intra- overlap
NCCL_API(ncclResult_t, ncclAllGatherCompTwoShot, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGatherCompTwoShot(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // optimize
  // Compress
 
  int* allIntraRank = (int*)malloc(comm->localRanks * sizeof(int));
  int* allInterRank = (int*)malloc(comm->nNodes * sizeof(int));
  int interCnt = 0, intraCnt = 0;
  for(int r = 0; r < comm->nRanks; r++){
    if(comm->rankToLocalRank[r] == comm->localRank) allInterRank[interCnt++] = r;
    if(comm->rankToNode[r] == comm->node) allIntraRank[intraCnt++] = r;
  }
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t totalSendBytes = (comm->nRanks + 1) * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aGMaxSendBytes;
  
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ? &recvbuff : &compBuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));

  if(mayUpdateBuff){
    aGMaxSendBytes = totalSendBytes;
    size_t compBuffBytes =  (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);
    // maxCompBuffBytes  = compBuffBytes;
    NCCLCHECK(allocAndRegCompBuff(comm, compBuffBytes));
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  
  void* sendCompbuff=compBuff;
  void* recvCompbuff=(char*)compBuff + compSendCount * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  // inter alltoall
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nNodes; r++){
    int peer = allInterRank[r];
    char* r_sendbuf =(char*) sendCompbuff;
    char* r_recvbuf =(char*) recvCompbuff + peer * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // intra alltoall
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->localRanks; r++){
    int peer = allIntraRank[r];
    if(peer == comm->rank) continue;
    for(int i = 0; i < comm->nNodes; i++){
      int sendLocation = allInterRank[i];
      int recvLocation = peer%comm->localRanks + i * comm->localRanks;
      char* r_sendbuf = (char*) recvCompbuff + sendLocation * compSendCount * ncclTypeSize(compDatatype);
      char* r_recvbuf = (char*) recvCompbuff + recvLocation * compSendCount * ncclTypeSize(compDatatype);
      NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, peer, comm, stream));
      NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, peer, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));


  // Free
  // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  free(allInterRank);
  free(allIntraRank);

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
  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));
  // NCCLCHECK(initCompressors());
  // NCCLCHECK(ncclCompress(sendbuff, sendcount, datatype, &sendCompbuff, &compSendCount, &compDatatype, 1, stream));

  NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype , &compSendCount, &compDatatype, 1, ncclCommOp_t::AllGather, stream));

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
  // NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllGather, stream));


  // Free
  CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff, stream));

  return ncclSuccess;
}

// max alltoall sendSize
__thread size_t a2AMaxSendSize = 0;

NCCL_API(ncclResult_t, ncclAllToAllComp, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAllToAllComp(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  
  // Compress
  
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * sendcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > a2AMaxSendSize;
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &compBuff, sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));

  if(mayUpdateBuff){
    a2AMaxSendSize = totalSendBytes;
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = compBuff;
  void* recvCompbuff = (char*)compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  //sendCompbuff + comm->nRanks * compSendCount * ncclTypeSize(ncclInt8)
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AlltoAll, stream));
    
  // Free
  // CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff,stream));
  
  return ncclSuccess;
}

// TODO comm- and comp- overlap
NCCL_API(ncclResult_t, ncclAlltoAllCompMultiStream, const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclAlltoAllCompMultiStream(const void* sendbuff, void* recvbuff, size_t sendcount,
  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

  // Compress
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));


  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));
  // NCCLCHECK(initCompressors());
  NCCLCHECK(initParallelComms(comm));

  // void** sendCompbuff;
  // void** recvCompbuff;
  // sendCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
  // recvCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));

  // void* sendCompbuff[8]={nullptr};
  // void* recvCompbuff[8]={nullptr};
  // new algo
  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[sendIdx], &compSendCount, &compDatatype, 1, parcomms[r].stream));
  //   CUDACHECK(cudaMallocAsync(&recvCompbuff[recvIdx], compSendCount * 1 * ncclTypeSize(compDatatype), parcomms[r].stream));
  // }
  // cudaEvent_t mainEvent;
  // CUDACHECK(cudaEventCreateWithFlags(&mainEvent, cudaEventDefault));
  // CUDACHECK(cudaEventRecord(mainEvent,stream));

  // for(int r=0;r<comm->nRanks;r++){
  //   CUDACHECK(cudaStreamWaitEvent(parcomms[r].stream, mainEvent, 0));
  // }
  // new algo
  // CUDACHECK(cudaMemcpyAsync((char*)recvbuff + comm->rank * sendcount * ncclTypeSize(datatype), (char*)sendbuff + comm->rank * sendcount * ncclTypeSize(datatype), 
    // sendcount *ncclTypeSize(datatype), cudaMemcpyDeviceToDevice, stream));

  // cudaMemcpyAsync(sendbuff, recvbuff, snedcount*ncclTypeSize(datatype), )

  void** sendCompbuff;
  void** recvCompbuff;
  // sendCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
  // recvCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
  sendCompbuff=(void**)calloc(2, sizeof(void*));
  recvCompbuff=(void**)calloc(2, sizeof(void*));
  // for(int commId =0; commId < 2;commId++){
  //   NCCLCHECK(ncclCompress((char*)sendbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[commId], &compSendCount, &compDatatype, comm->nRanks/2, parcomms[commId].stream));
  //   CUDACHECK(cudaMallocAsync(&recvCompbuff[commId], compSendCount * comm->nRanks/2 * ncclTypeSize(compDatatype), parcomms[commId].stream));
  // }


  for(int commId =0; commId < 2;commId++){
    // for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
    //   int sendIdx = (comm->rank + r)%comm->nRanks;
    //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
    //   NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[sendIdx], &compSendCount, &compDatatype, 1, parcomms[commId].stream));
    //   CUDACHECK(cudaMallocAsync(&recvCompbuff[recvIdx], compSendCount * 1 * ncclTypeSize(compDatatype), parcomms[commId].stream));
    // }
    NCCLCHECK(ncclCompress((char*)sendbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), &sendCompbuff[commId], 
              sendcount, datatype, &compSendCount, &compDatatype, comm->nRanks/2, ncclCommOp_t::AlltoAll, parcomms[commId].stream));

    // NCCLCHECK(ncclCompress((char*)sendbuff + r * sendcount * ncclTypeSize(datatype), &sendCompbuff[r], sendcount, datatype, &compSendCount, &compDatatype, 1, compstreams[r]));

    CUDACHECK(cudaMallocAsync(&recvCompbuff[commId], compSendCount * comm->nRanks/2 * ncclTypeSize(compDatatype), parcomms[commId].stream));
  }
  //   for(int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
  //     int sendIdx = (comm->rank + r)%comm->nRanks;
  //     int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //     NCCLCHECK(ncclCompress((char*)sendbuff + sendIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[sendIdx], &compSendCount, &compDatatype, 1, parcomms[r].stream));
  //     CUDACHECK(cudaMallocAsync(&recvCompbuff[recvIdx], compSendCount * 1 * ncclTypeSize(compDatatype), parcomms[r].stream));
  //   }
  // }
  // NCCLCHECK(ncclGroupStart());
  // for(int commId =0; commId < 2;commId++){
  //   for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
  //     int sendIdx = (comm->rank + r)%comm->nRanks;
  //     int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
      
  //     // char* r_sendbuf =sendCompbuff[sendIdx];
  //     // char* r_recvbuf =recvCompbuff[recvIdx];
  //     char* r_sendbuf =(char*) sendbuff + sendIdx * sendcount * ncclTypeSize(datatype);
  //     char* r_recvbuf =(char*) recvbuff + recvIdx * sendcount * ncclTypeSize(datatype);

  //     // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
  //     // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
  //     NCCLCHECK(ncclSend((void *) r_sendbuf, sendcount, datatype, sendIdx, parcomms[commId].subcomm, parcomms[commId].stream));
  //     NCCLCHECK(ncclRecv((void *) r_recvbuf, sendcount, datatype, recvIdx, parcomms[commId].subcomm, parcomms[commId].stream));
  //   }
  // }
  // NCCLCHECK(ncclGroupEnd());
  NCCLCHECK(ncclGroupStart());
  for(int commId =0; commId < 2;commId++){
    for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
          int sendIdx = (comm->rank + r)%comm->nRanks;
          int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
      
      // char* r_sendbuf =(char*) sendCompbuff[sendIdx];
      // char* r_recvbuf =(char*) recvCompbuff[recvIdx];
      char* r_sendbuf =(char*) sendCompbuff[commId] + r %( comm->nRanks/2 )* compSendCount * ncclTypeSize(compDatatype);
      char* r_recvbuf =(char*) recvCompbuff[commId] + r %( comm->nRanks/2 ) * compSendCount * ncclTypeSize(compDatatype);

      // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
      // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
      NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[commId].subcomm, parcomms[commId].stream));
      NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[commId].subcomm, parcomms[commId].stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

   for(int commId = 0; commId < 2; commId++){
    // for (int r= comm->nRanks/2 *commId;r<comm->nRanks/2*(commId + 1); r++){
    //   int sendIdx = (comm->rank + r)%comm->nRanks;
    //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
    //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[commId].stream));
    //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[commId].stream));
    //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[commId].stream));
    // }
    // NCCLCHECK(ncclDecompress(recvbuff, recvCompbuff, sendcount, datatype, compSendCount, compDatatype, comm->nRanks, stream));

    NCCLCHECK(ncclDecompress((char*)recvbuff + commId * comm->nRanks/2 * sendcount * ncclTypeSize(datatype), (void*)recvCompbuff[commId], 
              sendcount, datatype, compSendCount, compDatatype,  comm->nRanks/2, ncclCommOp_t::AlltoAll, parcomms[commId].stream));
    CUDACHECK(cudaFreeAsync(sendCompbuff[commId], parcomms[commId].stream));
    CUDACHECK(cudaFreeAsync(recvCompbuff[commId], parcomms[commId].stream));
  }
  
  // Decompress
  // new algo
  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[r].stream));
  // }
  
  for(int commId=0;commId<2;commId++){
    CUDACHECK(cudaEventRecord(parcomms[commId].event,parcomms[commId].stream));
    CUDACHECK(cudaStreamWaitEvent(stream, parcomms[commId].event, 0));
    // cudaStreamSynchronize(parcomms[r].stream);
  }


  // free(sendCompbuff);
  // free(recvCompbuff);


  // // old algo
  // for(int r=0;r<comm->nRanks;r++){
  //   NCCLCHECK(ncclCompress((char*)sendbuff + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[r], &compSendCount, &compDatatype, 1, streams[r]));
  //   CUDACHECK(cudaMallocAsync(&recvCompbuff[r], compSendCount * 1 * ncclTypeSize(compDatatype), streams[comm->rank]));
  //   // NCCLCHECK(ncclDecompress((void*)sendCompbuff[i], compSendCount, compDatatype, (char*)recvbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, streams[i]));
  // }
  // // old algo
  // NCCLCHECK(ncclGroupStart());
  // for (size_t r = 0; r < comm->nRanks ; r++){
  //   char* r_sendbuf =(char*) sendCompbuff[r];
  //   char* r_recvbuf =(char*) recvCompbuff[r];
  //   NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //   NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank], streams[comm->rank]));
  // }
  // NCCLCHECK(ncclGroupEnd());

  // // old algo
  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   // NCCLCHECK(ncclCompress((char*)sendbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[i], &compSendCount, &compDatatype,  1, streams[i]));
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[r], compSendCount, compDatatype, (char*)recvbuff + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, streams[comm->rank]));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], streams[r]));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], streams[comm->rank]));
  // }
  // for(int r=0;r<4;r++){
  //   // CUDACHECK(cudaStreamDestroy(streams+i));
  //   // CUDACHECK(cudaStreamDestroy(streams+i));
  //   CUDACHECK(cudaEventRecord(parcomms[r].event,parcomms[r].stream));
  //   CUDACHECK(cudaStreamWaitEvent(stream, parcomms[r].event, 0));
  // }


  // two rank tests
    // NCCLCHECK(ncclGroupStart());
    // for(int r = 0; r< 4;r++){
    //   int sendIdx = (comm->rank + r)%4;
    //   int recvIdx = (comm->rank - r + 4)%4;

    //   // char* r_sendbuf =(char*) sendbuff + sendIdx * sendcount * ncclTypeSize(datatype);
    //   // char* r_recvbuf =(char*) recvbuff + recvIdx * sendcount * ncclTypeSize(datatype);
    //         // NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
    //         // NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank], streams[comm->rank]));
    //         // NCCLCHECK(ncclGroupStart());
    //     char* r_sendbuf =(char*) sendCompbuff[sendIdx];
    //     char* r_recvbuf =(char*) recvCompbuff[recvIdx];

    //   // multi
    //   // NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, sendIdx, parcomms[r].subcomm, parcomms[r].stream));
    //   // NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, recvIdx, parcomms[r].subcomm, parcomms[r].stream));
    // }
    // NCCLCHECK(ncclGroupEnd());

  // cudaEvent_t mainevent;
  // CUDACHECK(cudaEventCreateWithFlags(&mainevent, cudaEventDefault));
  // CUDACHECK(cudaEventRecord(mainevent,stream));

  // for(int r=0;r<comm->nRanks;r++){
  //   // CUDACHECK(cudaFree(sendCompbuff+i))
  //   CUDACHECK(cudaStreamWaitEvent(parcomms[r].stream, mainevent, 0));
  // }



  // for(int r=0;r<comm->nRanks;r++){
  //   int sendIdx = (comm->rank + r)%comm->nRanks;
  //   int recvIdx = (comm->rank - r + comm->nRanks)%comm->nRanks;
  //   // NCCLCHECK(ncclCompress((char*)sendbuff + i * sendcount * ncclTypeSize(datatype), sendcount, datatype, &sendCompbuff[i], &compSendCount, &compDatatype,  1, streams[i]));
  //   NCCLCHECK(ncclDecompress((void*)recvCompbuff[recvIdx], compSendCount, compDatatype, (char*)recvbuff + recvIdx * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(sendCompbuff[sendIdx], parcomms[r].stream));
  //   CUDACHECK(cudaFreeAsync(recvCompbuff[recvIdx], parcomms[r].stream));
  // }


    // NCCLCHECK(ncclGroupStart());
    // for(int r = 0; r< 4;r++){

    //   char* r_sendbuf =(char*) sendbuff + r * sendcount * ncclTypeSize(datatype);
    //   char* r_recvbuf =(char*) recvbuff + r * sendcount * ncclTypeSize(datatype);

    //   // multi
    //   NCCLCHECK(ncclSend((void *) r_sendbuf, sendcount, datatype, r, parcomms[comm->rank].subcomm, parcomms[comm->rank].stream));
    //   NCCLCHECK(ncclRecv((void *) r_recvbuf, sendcount, datatype, r, parcomms[r].subcomm, parcomms[r].stream));
    
    // }
    // NCCLCHECK(ncclGroupEnd());



  // for(int i=0;i<comm->nRanks;i++){
  //   // CUDACHECK(cudaFree(sendCompbuff+i))
  //   CUDACHECK(cudaEventRecord(events[i],streams[i]));
  //   CUDACHECK(cudaStreamWaitEvent(stream, events[i], 0));
  // }


  // NCCLCHECK(ncclGroupStart());
  //   for (size_t r = 0; r < comm->nRanks ; r++){
  //     // char* r_sendbuf =(char*) sendbuff + r * sendcount*ncclTypeSize(datatype);
  //     // char* r_recvbuf =(char*) recvbuff + r * sendcount*ncclTypeSize(datatype);
  //     char* r_sendbuf =(char*) sendCompbuff[r];
  //     char* r_recvbuf =(char*) sendCompbuff[r];
  //       // NCCLCHECK(ncclGroupStart());
  //       // NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //       // NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[r], streams[r]));
  //       NCCLCHECK(ncclSend((void *)r_sendbuf, compSendCount, compDatatype, r, subcomms[comm->rank * comm->nRanks + r], streams[r]));
  //       NCCLCHECK(ncclRecv((void *)r_recvbuf, compSendCount, compDatatype, r, subcomms[comm->rank * comm->nRanks + r], streams[r]));
  //       // NCCLCHECK(ncclGroupEnd());
  //       // CUDACHECK(cudaEventRecord(events[2 * r],streams[2 * r]));
  //       // CUDACHECK(cudaEventRecord(events[2 * r + 1],streams[2 * r + 1]));
  //       // CUDACHECK(cudaStreamWaitEvent(stream, events[2 * r], 0));
  //       // CUDACHECK(cudaStreamWaitEvent(stream, events[2 * r + 1], 0));
  //   }
  //   NCCLCHECK(ncclGroupEnd());


    // for(int r=0;r<comm->nRanks;r++){
    //   // CUDACHECK(cudaStreamDestroy(streams+i));
    //   // CUDACHECK(cudaStreamDestroy(streams+i));
    //   CUDACHECK(cudaEventRecord(events[r],streams[r]));
    //   CUDACHECK(cudaStreamWaitEvent(stream, events[r], 0));
    // }





  // NCCLCHECK(ncclDecompress((void*)sendCompbuff, compSendCount, compDatatype, recvbuff, sendcount, datatype, comm->nRanks, stream));

  // Free
  // for(int r=0;r<4;r++){
  //   CUDACHECK(cudaStreamSynchronize(parcomms[r].stream));
  // }

  // for(int r=0;r<comm->nRanks;r++){
  //     // CUDACHECK(cudaStreamDestroy(streams+i));
  //     // CUDACHECK(cudaStreamDestroy(streams+i));
  //     CUDACHECK(cudaEventRecord(parcomms[r].event,parcomms[r].stream));
  //     CUDACHECK(cudaStreamWaitEvent(stream, parcomms[r].event, 0));
  // }

  return ncclSuccess;
}
// max reduceScatter sendSize


// max reduceScatter sendSize
__thread size_t aRMaxSendSize = 0;

NCCL_API(ncclResult_t, ncclAllReduceCompOneShot, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOneShot(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  // void* recvTempbuff = nullptr;
  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t numChunks = comm->nRanks;
  // CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), stream));
  // Compress
  void* sendCompbuff=nullptr;
  void* recvCompbuff=nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  // NCCLCHECK(ncclCompress(sendbuff, chunkCount, datatype, &sendCompbuff, &compSendCount, &compDatatype, numChunks, stream));
  NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype,  &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  CUDACHECK(cudaMallocAsync((void**)&recvCompbuff,  comm->nRanks * numChunks * compSendCount * ncclTypeSize(compDatatype), stream));

  //Gather

  // broadcast based - allchunk 
  // NCCLCHECK(ncclGroupStart());
  // for(int r = 0; r < comm->nRanks; r++){
  //   char* r_recvbuf = (char*)recvCompbuff + r * compSendCount * ncclTypeSize(compDatatype);
  //   NCCLCHECK(ncclBroadcast((void*)sendCompbuff, (void*)r_recvbuf, compSendCount, compDatatype, r, comm, stream));
  // }
  // NCCLCHECK(ncclGroupEnd());

  // broadcast based - chunk parallel
  // for(int r = 0; r < comm->nRanks; r++){
  //   NCCLCHECK(ncclGroupStart());
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf = (char*)sendCompbuff + chunkIdx * compSendCount * ncclTypeSize(compDatatype);
  //     char* r_recvbuf = (char*)recvCompbuff + (r * comm->nRanks + chunkIdx) * compSendCount * ncclTypeSize(compDatatype);
  //     NCCLCHECK(ncclBroadcast((void*)r_sendbuf, (void*)r_recvbuf, compSendCount, compDatatype, r, comm, stream));
  //   }
  //   NCCLCHECK(ncclGroupEnd());
  // }

  // NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvTempbuff, chunkCount, datatype, comm->nRanks * numChunks, stream));

  




  // P2P based - allchunk
  // in RTX 4090 platform it is faster than broadcast based and p2p chunk parallel 50% 
  // size 1K ~ 1M
  NCCLCHECK(ncclGroupStart());

  for(int r = 0; r < comm->nRanks; r++){
    char* r_recvbuf = (char*)recvCompbuff + r * numChunks * compSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclSend(sendCompbuff, numChunks * compSendCount, compDatatype, r, comm, stream));
    NCCLCHECK(ncclRecv((void*)r_recvbuf, numChunks * compSendCount, compDatatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());


  // p2p based - chunk parallel
  // for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //   char* r_sendbuf =  (char*)sendbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     for(int r = 0; r < comm->nRanks; r++){
  //       char* r_recvbuf = (char*)recvTempbuff + (r * comm->nRanks + chunkIdx) * chunkCount * ncclTypeSize(datatype);
  //       NCCLCHECK(ncclGroupStart());
  //       NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, r, comm, stream));
  //       NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  //       NCCLCHECK(ncclGroupEnd());
  //     }
  // }
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, numChunks * compSendCount, compDatatype, numChunks * chunkCount, datatype, comm->nRanks,
  ncclCommOp_t::AllReduce, stream));
  // NCCLCHECK(ncclDecompress(recvTempbuff, (void*)recvCompbuff, numChunks * chunkCount, datatype, compSendCount, compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));


  // // Reduce chunk
  // NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));
  
  CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff,stream));


  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShotR0, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShotR0(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {


  size_t chunkCount = DIVUP(count, comm->nRanks);
  size_t numChunks = comm->nRanks;
  void* recvTempbuff = nullptr;
  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));
  // NCCLCHECK(initCompressors());
  if(comm->rank==0)
    CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * numChunks * chunkCount * ncclTypeSize(datatype), stream));  

  // Compress
  void* sendCompbuff=nullptr;
  void* recvCompbuff=nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  // NCCLCHECK(ncclCompress(sendbuff, chunkCount, datatype, &sendCompbuff, &compSendCount, &compDatatype, numChunks, stream));
  NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, numChunks, ncclCommOp_t::AllReduce, stream));

  if(comm->rank ==0){
    CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, comm->nRanks * numChunks * compSendCount * ncclTypeSize(compDatatype), stream));
  } else {
    CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, numChunks * compSendCount * ncclTypeSize(compDatatype), stream));
  }


  // Gather 
  // p2p base - allchunk
  // in RTX 4090 platform it is faster than p2p chunk parallel 10% 
  // size 1M ~ 32M
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclSend(sendCompbuff, compSendCount * numChunks, compDatatype, 0, comm, stream));
  if (comm->rank == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      NCCLCHECK(ncclRecv(((char*)recvCompbuff) + r * numChunks * compSendCount * ncclTypeSize(compDatatype), compSendCount * numChunks, compDatatype, r, comm, stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());


  // p2p base - chunk parallel
  // NCCLCHECK(ncclGroupStart());
  // for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf =  (char*)sendbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, 0, comm, stream));
  // }
  // if(comm->rank == 0){
  //   for(int r = 0; r < comm->nRanks; r++){
  //     for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //       char* r_recvbuf = (char*)recvTempbuff + (r * comm->nRanks + chunkIdx) * chunkCount * ncclTypeSize(datatype);
  //       NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, r, comm, stream));
  //     }
  //   }
  // }
  // NCCLCHECK(ncclGroupEnd());

  // Reduce Chunk
  if(comm->rank == 0) {
    // NCCLCHECK(ncclDecompress((void*)recvCompbuff, compSendCount, compDatatype, recvTempbuff, chunkCount, datatype, comm->nRanks * numChunks, stream));
    NCCLCHECK(ncclDecompress(recvTempbuff, (void*)recvCompbuff, chunkCount, datatype, compSendCount, compDatatype, comm->nRanks * numChunks, ncclCommOp_t::AllReduce, stream));
    
    NCCLCHECK(ncclReduceChunk(recvTempbuff, numChunks * chunkCount, recvbuff, datatype, comm->nRanks, stream));
    // recompress
    NCCLCHECK(ncclCompress(recvbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, numChunks, ncclCommOp_t::AllReduce, stream));
  }

  // Scatter

  // broadcast base - allchunk
  // NCCLCHECK(ncclGroupStart());
  // NCCLCHECK(ncclBroadcast(recvbuff, recvbuff, count, datatype, 0, comm, stream));
  // NCCLCHECK(ncclGroupEnd());

  // broadcast base - chunk parallel
  // in RTX 4090 platform it is faster than p2p based and all chunk broadcast 30% 
  // size 1M ~ 32M
  // NCCLCHECK(ncclGroupStart());
  // for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //   char* r_sendbuf = (char*)sendCompbuff + chunkIdx * compSendCount * ncclTypeSize(compDatatype);
  //   char* r_recvbuf = (char*)recvCompbuff + chunkIdx * compSendCount * ncclTypeSize(compDatatype);
  //   NCCLCHECK(ncclBroadcast((void*)r_sendbuf, (void*)r_recvbuf, compSendCount, compDatatype, 0, comm, stream));
  // }
  // NCCLCHECK(ncclGroupEnd());




  // p2p base - allchunk
  if(comm->rank == 0){
    for(int r = 1; r< comm->nRanks; r++){
      NCCLCHECK(ncclSend(((char*)recvbuff), count, datatype, r, comm, stream));
    }
  }else{
    NCCLCHECK(ncclRecv(recvbuff, count, datatype, 0, comm, stream));
  }

  // p2p base - chunk parallel
  // if(comm->rank == 0){
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_sendbuf =  (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     for(int r = 1; r < comm->nRanks; r++){
  //       NCCLCHECK(ncclSend((void*)r_sendbuf, chunkCount, datatype, r, comm, stream));
  //     }
  //   }
  // }else{
  //   for(int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++){
  //     char* r_recvbuf =  (char*)recvbuff + chunkIdx * chunkCount * ncclTypeSize(datatype);
  //     NCCLCHECK(ncclRecv((void*)r_recvbuf, chunkCount, datatype, 0, comm, stream));
  //   }
  // }
  // decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, compSendCount, compDatatype, numChunks, ncclCommOp_t::AllReduce, stream));

  // Free
  CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  CUDACHECK(cudaFreeAsync(recvCompbuff,stream));
  if(comm->rank == 0)
    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompTwoShotAll, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompTwoShotAll(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  size_t chunkCount = DIVUP(count, comm->nRanks);
 
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  size_t compSendCount;
  ncclDataType_t compDatatype;
  size_t totalSendBytes = 2 * count * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > aRMaxSendSize;
 
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  NCCLCHECK(ncclCompress(sendbuff, mayUpdateBuff ?  &recvbuff : &compBuff, chunkCount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));
  
  if(mayUpdateBuff){
    aRMaxSendSize = totalSendBytes;
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, recvbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
  }
  void* sendCompbuff = compBuff;
  void* recvCompbuff = (char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  //sendCompbuff + comm->nRanks * compSendCount * ncclTypeSize(ncclInt8)
  
  NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
  // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, compSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->nRanks,
                        ncclCommOp_t::AllReduce, stream));
  // Scatter
  // p2p base - chunk 
  // in RTX 4090 platform it is faster than broadcast based 20% 
  // size 1M ~ 32M
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r <  comm->nRanks; r++){
    char* r_sendbuf = (char*)sendCompbuff;
    char* r_recvbuf = (char*)recvCompbuff + r * reCompSendCount * ncclTypeSize(reCompDatatype);
      NCCLCHECK(ncclSend((void*)r_sendbuf, reCompSendCount, reCompDatatype, r, comm, stream));
      NCCLCHECK(ncclRecv((void*)r_recvbuf, reCompSendCount, reCompDatatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());

  // Decompress
  NCCLCHECK(ncclDecompress(recvbuff, (void*)recvCompbuff, chunkCount, datatype, reCompSendCount, reCompDatatype, comm->nRanks, ncclCommOp_t::AllReduce, stream));

  // Frees
  // CUDACHECK(cudaFreeAsync(sendCompbuff,stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff,stream));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompRing, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompRing(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  size_t chunkCount = count / comm->nRanks;
  // NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));
  // NCCLCHECK(initCompressors());

  char* r_recvbuf = (char*) recvbuff + comm->rank * chunkCount * ncclTypeSize(datatype);

  NCCLCHECK(ncclReduceScatterComp(sendbuff, r_recvbuf, chunkCount, datatype, op, comm, stream));

  NCCLCHECK(ncclAllGatherComp(r_recvbuf, recvbuff, chunkCount, datatype, comm, stream));
  
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduceCompOptim, const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceCompOptim(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
  if(count * ncclTypeSize(datatype) < (size_t)1024 * 1024){
    NCCLCHECK(ncclAllReduceOneShot(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  else if(count * ncclTypeSize(datatype) < (size_t)1024 * 1024 * 32){
    NCCLCHECK(ncclAllReduceCompTwoShotAll(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }else{
    NCCLCHECK(ncclAllReduceCompRing(sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return ncclSuccess;
}
__thread size_t rSMaxSendSize = 0;

NCCL_API(ncclResult_t, ncclReduceScatterCompOneShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompOneShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){

  if(datatype == ncclDataType_t::ncclFloat16 || datatype == ncclDataType_t::ncclBfloat16){
    void* recvTempbuff = nullptr;
    // CUDACHECK(cudaMallocAsync((void**)&recvTempbuff, comm->nRanks * recvcount * ncclTypeSize(datatype), stream));
    CUDACHECK(cudaMallocFromPoolAsync((void**)&recvTempbuff, comm->nRanks * recvcount * ncclTypeSize(datatype), compMemPool[comm->cudaDev % compMemPoolCnt], stream));
    // Gather
    NCCLCHECK(ncclAllToAllComp(sendbuff, recvTempbuff, recvcount, datatype, comm, stream));
    // Reduce
    NCCLCHECK(ncclReduceChunk(recvTempbuff, recvcount, recvbuff, datatype, comm->nRanks, stream));

    CUDACHECK(cudaFreeAsync(recvTempbuff, stream));
  }
  else if(datatype == ncclDataType_t::ncclFloat32){

   

    size_t compSendCount;
    ncclDataType_t compDatatype;
    
    CUDACHECK(cudaSetDevice(comm->cudaDev));
    size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
    bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

    if(mayUpdateBuff){
      rSMaxSendSize = totalSendBytes;
      void* tempCompbuff = nullptr;
      NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
        ncclCommOp_t::ReduceScatter, stream));
      size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
      allocAndRegCompBuff(comm, compBuffBytes);
      CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
      CUDACHECK(cudaDeviceSynchronize());
      CUDACHECK(cudaFree(tempCompbuff));
    } else {
      NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
        ncclCommOp_t::ReduceScatter, stream));
    }

    void* sendCompbuff = compBuff;
    void* recvCompbuff =(char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    
    // NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));
    NCCLCHECK(ncclAllToAll((void*)sendCompbuff, (void*)recvCompbuff, compSendCount, compDatatype, comm, stream));

    // DecompReduce
    // NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
    //                     ncclCommOp_t::ReduceScatter, stream));
    NCCLCHECK(ncclDecompressReduce((void*)recvbuff, (void*)recvCompbuff, compSendCount, compDatatype, recvcount, datatype, comm->nRanks,
                        ncclCommOp_t::ReduceScatter, stream));

    
    // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
    // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  }
  
  return ncclSuccess;
}

// TODO inter- and intra- overlap
NCCL_API(ncclResult_t, ncclReduceScatterCompTwoShot, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterCompTwoShot(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream){
    // there could do inter and intra optimize, multiComm and multiStream

  // get intraRanks and interRanks
  int* allIntraRank = (int*)malloc(comm->localRanks * sizeof(int));
  int* allInterRank = (int*)malloc(comm->nNodes * sizeof(int));
  int interCnt = 0, intraCnt = 0;
  for(int r = 0; r < comm->nRanks; r++){
    if(comm->rankToLocalRank[r] == comm->localRank) allInterRank[interCnt++] = r;
    if(comm->rankToNode[r] == comm->node) allIntraRank[intraCnt++] = r;
  }
  // void* sendCompbuff = nullptr;
  // void* recvCompbuff = nullptr;
  size_t compSendCount;
  ncclDataType_t compDatatype;
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t totalSendBytes = 2 * comm->nRanks * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
    size_t compBuffBytes = 2 * compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
  }

  void* sendCompbuff = compBuff;
  void* recvCompbuff =(char*) compBuff + compSendCount * comm->nRanks * ncclTypeSize(compDatatype);
  // reuse buff may have some wrong, some data may not send/recv sometimes
  // swizzle and quan
  // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, ncclCommOp_t::ReduceScatter, stream));

  // CUDACHECK(cudaMallocAsync((void**)&recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), stream));
  // intra alltoall
  size_t intraSendCount = compSendCount * comm->nNodes;
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->localRanks ; r++){
    int peer = allIntraRank[r];
    char* r_sendbuf =(char*) sendCompbuff + r * intraSendCount * ncclTypeSize(compDatatype);
    char* r_recvbuf =(char*) recvCompbuff + r * intraSendCount * ncclTypeSize(compDatatype);
    NCCLCHECK(ncclRecv((void *)r_recvbuf, intraSendCount, compDatatype, peer, comm, stream));
    NCCLCHECK(ncclSend((void *)r_sendbuf, intraSendCount, compDatatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
   
  size_t reCompSendCount;
  ncclDataType_t reCompDatatype;
    // DecompReduceComp
  NCCLCHECK(ncclDecompReduceComp((void*)recvCompbuff, &sendCompbuff, intraSendCount, compDatatype, &reCompSendCount, &reCompDatatype, comm->localRanks,
                      ncclCommOp_t::ReduceScatter_Inter, stream));

     
    // inter alltoall
  size_t interSendCount = reCompSendCount / comm->nNodes;
  NCCLCHECK(ncclGroupStart());
  for(int r = 0; r < comm->nNodes; r++){
    int peer = allInterRank[r];
    char* r_sendbuf =(char*) sendCompbuff + r * interSendCount * ncclTypeSize(reCompDatatype);
    char* r_recvbuf =(char*) recvCompbuff + r * interSendCount * ncclTypeSize(reCompDatatype);
    NCCLCHECK(ncclRecv((void *)r_recvbuf, interSendCount, reCompDatatype, peer, comm, stream));
    NCCLCHECK(ncclSend((void *)r_sendbuf, interSendCount, reCompDatatype, peer, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
    
  // DecompReduce
  NCCLCHECK(ncclDecompressReduce((void*)recvbuff, recvCompbuff, interSendCount, reCompDatatype, recvcount, datatype, comm->nNodes,
                        ncclCommOp_t::ReduceScatter_Inter, stream));
  

  // CUDACHECK(cudaFreeAsync(sendCompbuff, stream));
  // CUDACHECK(cudaFreeAsync(recvCompbuff, stream));
  free(allInterRank);
  free(allIntraRank);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatterComp, const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatterComp(const void* sendbuff, void* recvbuff, size_t recvcount,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  int rightRank = (comm->rank + 1) % comm->nRanks;
  int leftRank = (comm->rank - 1 + comm->nRanks) % comm->nRanks;
  // INFO(NCCL_INIT, "coccl ReduceScatter comp ring");


  size_t chunkBytes = recvcount * ncclTypeSize(datatype);
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  size_t compSendCount;
  ncclDataType_t compDatatype;
  size_t totalSendBytes = (2 + comm->nRanks) * recvcount * ncclTypeSize(datatype);
  bool mayUpdateBuff = compBuffHandle == nullptr || totalSendBytes > rSMaxSendSize;

  if(mayUpdateBuff){
    rSMaxSendSize = totalSendBytes;
    void* tempCompbuff = nullptr;
    NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
    size_t compBuffBytes = compSendCount * (comm->nRanks + 2) * ncclTypeSize(compDatatype);
    allocAndRegCompBuff(comm, compBuffBytes);
    CUDACHECK(cudaMemcpy(compBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(tempCompbuff));
  } else {
    NCCLCHECK(ncclCompress(sendbuff, &compBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    ncclCommOp_t::ReduceScatter, stream));
  }
  void* reduceSendbuf = (char*) compBuff + comm->nRanks * compSendCount * ncclTypeSize(compDatatype);
  void* reduceRecvbuf = (char*) compBuff + (comm->nRanks + 1) * compSendCount * ncclTypeSize(compDatatype);

  for (int r = comm->nRanks - 1; r >= 0; r--) {
    // Ring step 0
    // compress - recv -  send
    int sendIdx = (comm->rank + r) % comm->nRanks;
    int recvIdx = (comm->rank + (r - 1) + comm->nRanks) % comm->nRanks;

    CUDACHECK(cudaMemcpyAsync(reduceSendbuf, (char*)compBuff + sendIdx * compSendCount * ncclTypeSize(compDatatype), 
                                          compSendCount * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice, stream));                            

    if(r == comm->nRanks - 1){
   
      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());

    } else if(r > 0) {
      // Ring step 1 ~ N - 2
      // DecompReduceComp
      NCCLCHECK(ncclDecompReduceComp((void*)reduceSendbuf, (void**)&reduceSendbuf, compSendCount, compDatatype, &compSendCount, &compDatatype, 2,
                          ncclCommOp_t::ReduceScatter, stream));

      NCCLCHECK(ncclGroupStart());
      NCCLCHECK(ncclRecv((void*)reduceRecvbuf, compSendCount, compDatatype, leftRank, comm, stream));
      NCCLCHECK(ncclSend((void*)reduceSendbuf, compSendCount, compDatatype, rightRank, comm, stream));
      NCCLCHECK(ncclGroupEnd());
    } else {
      // Ring step N - 1
      // decompress - reduce
      NCCLCHECK(ncclDecompressReduce((void*)recvbuff, reduceSendbuf, compSendCount, compDatatype, recvcount, datatype, 2,
                        ncclCommOp_t::ReduceScatter, stream));
    }
  }
  

  return ncclSuccess;
}



// NCCL_API(ncclResult_t, ncclAlltoAllCompSingleCommMulitComp, const void* sendbuff, void* recvbuff, size_t sendcount,
//   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
// ncclResult_t  ncclAlltoAllCompSingleCommMulitComp(const void* sendbuff, void* recvbuff, size_t sendcount,
//   ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {

//   // Compress
//   size_t compSendCount;
//   ncclDataType_t compDatatype;
//   CUDACHECK(cudaSetDevice(comm->cudaDev));
//   NCCLCHECK(initCompMemPool(comm->cudaDev, comm->localRanks));


//   if(compstreams == nullptr){
//     compstreams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * comm->nRanks);
//     compevents = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * comm->nRanks);
//     for(int i=0;i<comm->nRanks;i++){
//       CUDACHECK(cudaStreamCreateWithFlags(compstreams+i, cudaStreamNonBlocking));
//       CUDACHECK(cudaEventCreateWithFlags(compevents+i, cudaEventDefault));
//     }
//   }
//   void** sendCompbuff;
//   // void** recvCompbuff;
//   sendCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
//   // recvCompbuff=(void**)calloc(comm->nRanks, sizeof(void*));
//   // void* sendCompbuff[8]={nullptr};
  
//   for(int r=0;r<comm->nRanks;r++){
//     NCCLCHECK(ncclCompress((char*)sendbuff + r * sendcount * ncclTypeSize(datatype), &sendCompbuff[r], sendcount, datatype, &compSendCount, &compDatatype, 1, compstreams[r]));
//     // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, sendcount, datatype &compSendCount, &compDatatype, comm->nRanks, stream));

//   }

//   NCCLCHECK(ncclGroupStart());
//   for(int r = 0; r< comm->nRanks;r++){
//     char* r_sendbuf =(char*) sendCompbuff[r];
//     char* r_recvbuf =(char*) sendCompbuff[r];
//     // multi
//     NCCLCHECK(ncclSend((void *) r_sendbuf, compSendCount, compDatatype, r, comm, compstreams[r]));
//     NCCLCHECK(ncclRecv((void *) r_recvbuf, compSendCount, compDatatype, r, comm, compstreams[r]));
//   }
//   NCCLCHECK(ncclGroupEnd());

//   for(int r=0;r<comm->nRanks;r++){
//     // NCCLCHECK(ncclDecompress((void*)sendCompbuff[r], compSendCount, compDatatype, (char*)recvbuff + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, 1, compstreams[r]));
//     NCCLCHECK(ncclDecompress((char*)recvbuff + r * sendcount * ncclTypeSize(datatype), (void*)sendCompbuff[r], sendcount, datatype, compSendCount, compDatatype, 1, compstreams[r]));

//     CUDACHECK(cudaFreeAsync(sendCompbuff[r], compstreams[r]));
//   }
//   for(int r=0;r<comm->nRanks;r++){
//     CUDACHECK(cudaEventRecord(compevents[r],compstreams[r]));
//     CUDACHECK(cudaStreamWaitEvent(stream, compevents[r], 0));
//   }
//   free(sendCompbuff);


//   return ncclSuccess;
// }



// allreduce
// cudaStreamSynchronize(stream);

// if(sendbuff !=recvbuff && comm->rank == 0)
// {
//   float *send = (float*)malloc(count * sizeof(float));
//   float *recv = (float*)malloc(count * sizeof(float));
//   // uint8_t *recvT = (uint8_t*)malloc(compSendCount*comm->nRanks * ncclTypeSize(compDatatype));
//   cudaMemcpyAsync(recv, recvbuff, count * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   // cudaMemcpyAsync(recvT, recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(send, sendbuff, count * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaStreamSynchronize(stream);

//   // CUDACHECK(cudaDeviceSynchronize());
//   for(int r=0;r<comm->nRanks;r++){
          
//         for(int i=0;i<chunkCount;i++){
//             // const float min_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount));
//             // const float max_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount + sizeof(float)));
//             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f reduced %f",comm->rank, r ,i ,send[r*chunkCount+i], recv[r*chunkCount+i]);
//             // else
//             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f",comm->rank, r ,i , send[r*recvcount+i], recvT[r*recvcount+i]);
//         }

//         INFO(NCCL_INIT, "chunkid %d: ", r);
//   }
//   INFO(NCCL_INIT,"rank: %d", comm->rank);
//   // free(quan);
//   free(send);
//   // free(recvT);
//   free(recv);
// }

// reduce scatter comp
// if(sendbuff !=recvbuff)
  // {

  // float *send = (float*)malloc(recvcount*comm->nRanks * sizeof(float));
  // float *recv = (float*)malloc(recvcount* sizeof(float));
  // uint8_t *recvT = (uint8_t*)malloc(compSendCount*comm->nRanks * ncclTypeSize(compDatatype));
  // cudaMemcpyAsync(recv, recvbuff, recvcount * sizeof(float), cudaMemcpyDeviceToHost,stream);
  // cudaMemcpyAsync(recvT, recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToHost,stream);
  // cudaMemcpyAsync(send, sendbuff, recvcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
  // cudaStreamSynchronize(stream);

  // // CUDACHECK(cudaDeviceSynchronize());
  //  for(int r=0;r<comm->nRanks;r++){
        
  //       for(int i=0;i<recvcount;i++){
  //           if(r==comm->rank){
  //           // const float min_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount));
  //           // const float max_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount + sizeof(float)));
  //           INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f min %f max %f tmprecv %u reduced %f",comm->rank, r ,i ,send[r*recvcount+i], *(reinterpret_cast<float *>(recvT + r * compSendCount)), 
  //           *(reinterpret_cast<float *>(recvT + r * compSendCount + sizeof(float))), recvT[r*compSendCount+32+i], recv[i]);
  //         }
  //           // else
  //           // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f",comm->rank, r ,i , send[r*recvcount+i], recvT[r*recvcount+i]);
  //       }

  //       INFO(NCCL_INIT, "chunkid %d: ", r);
  // }
  // INFO(NCCL_INIT,"rank: %d", comm->rank);
  // // free(quan);
  // free(send);
  // free(recvT);
  // free(recv);
  // }

// reduce scatter temp
// if(sendbuff !=recvbuff)
//   {

//   float *send = (float*)malloc(recvcount*comm->nRanks * sizeof(float));
//   float *recv = (float*)malloc(recvcount* sizeof(float));
//   float *recvT = (float*)malloc(recvcount*comm->nRanks * sizeof(float));

//   cudaMemcpyAsync(recv, recvbuff, recvcount * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(recvT, recvTempbuff, recvcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(send, sendbuff, recvcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaStreamSynchronize(stream);

//   // CUDACHECK(cudaDeviceSynchronize());
//    for(int r=0;r<comm->nRanks;r++){
//         // const float min_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * chunk_offset));
//         // const float max_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * chunk_offset + sizeof(T)));
//         for(int i=0;i<recvcount;i++){
//             if(r==comm->rank)
//             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f reduced %f",comm->rank, r ,i ,send[r*recvcount+i],recvT[r*recvcount+i], recv[i]);
//             // else
//             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f",comm->rank, r ,i , send[r*recvcount+i], recvT[r*recvcount+i]);
//         }

//         INFO(NCCL_INIT, "chunkid %d: ", r);
//   }
//   INFO(NCCL_INIT,"rank: %d", comm->rank);
//   // free(quan);
//   free(send);
//   free(recvT);
//   free(recv);
//   }

// NCCL_API(ncclResult_t, ncclSendComp, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
//     ncclComm_t comm, cudaStream_t stream);
// ncclResult_t ncclSendComp(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
//     ncclComm_t comm, cudaStream_t stream) {
//   NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
//   NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

//   struct ncclInfo info = { ncclFuncSend, "Send",
//     NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
//     1, 1 };
//   ncclResult_t ret;
//   NCCLCHECK(ncclGroupStart());
//   NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
// exit:
//   NCCLCHECK(ncclGroupEnd());
//   return ret;
// }

// NCCL_API(ncclResult_t, ncclRecvDecomp, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
//     ncclComm_t comm, cudaStream_t stream);
// ncclResult_t ncclRecvDecomp(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
//     ncclComm_t comm, cudaStream_t stream) {
//   NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
//   NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

//   struct ncclInfo info = { ncclFuncRecv, "Recv",
//     NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
//     1, 1 };
//   ncclResult_t ret;
//   NCCLCHECK(ncclGroupStart());
//   NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
// exit:
//   NCCLCHECK(ncclGroupEnd());
//   return ret;
// }



// alltoall comp
// if(sendbuff !=recvbuff)
//   {
//   uint8_t *recvquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(uint8_t));
//   uint8_t *sendquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(uint8_t));
//   float *send = (float*)malloc(sendcount*comm->nRanks * sizeof(float));
//   float *recv = (float*)malloc(sendcount*comm->nRanks * sizeof(float));
//   cudaMemcpyAsync(recv, recvbuff, sendcount*comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(send, sendbuff, sendcount*comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(recvquan, recvCompbuff, compSendCount * comm->nRanks * sizeof(uint8_t), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(sendquan, sendCompbuff, compSendCount * comm->nRanks * sizeof(uint8_t), cudaMemcpyDeviceToHost,stream);
//   cudaStreamSynchronize(stream);
//   // CUDACHECK(cudaDeviceSynchronize());
//    for(int r=0;r<comm->nRanks;r++){
//         INFO(NCCL_INIT, "min:%f ", *(reinterpret_cast<float *>(recvquan + r * compSendCount)));
//         INFO(NCCL_INIT, "max:%f ", *(reinterpret_cast<float *>(recvquan + r * compSendCount + sizeof(float))));
//         // const float min_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * chunk_offset));
//         // const float max_ = __load_as_float(reinterpret_cast<T *>(compbuff + idy * chunk_offset + sizeof(T)));
//         for(int i=0;i<sendcount;i++){
//             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f sendquan %u recvquan %u dequan %f", 
//               comm->rank, r, i, send[r*sendcount+i], sendquan[r*compSendCount+32+i],recvquan[r*compSendCount+32+i], recv[r*sendcount+i]);

//         }
//         // INFO(NCCL_INIT, "chunkid %d: ", r);
//     }
//   // INFO(NCCL_INIT,"rank: %d", comm->rank);
//   free(recvquan);
//   free(sendquan);
//   free(send);
//   free(recv);
//   }

  // size_t freemem; 
  // cudaMemGetInfo(&freemem, NULL);
  // INFO(NCCL_INIT, "nccl rank %d, cudaDev %d, sendcount %ld, nranks %d, sendbytes %ld, compSendCount %ld, compbytes %ld, comp rate %f, send comp buff: %p, recv comp buff: %p, remaining GMem Size: %ld",
  // comm->rank, comm->cudaDev,sendcount, comm->nRanks, sendcount*comm->nRanks*ncclTypeSize(datatype), compSendCount, compSendCount*comm->nRanks * ncclTypeSize(compDatatype),
  // sendcount*comm->nRanks*ncclTypeSize(datatype)/float(compSendCount* comm->nRanks * ncclTypeSize(compDatatype)), sendCompbuff, recvCompbuff, freemem);


  // void* ttt = nullptr;
  // void* rrr = nullptr;
  // CUDACHECK(cudaSetDevice(comm->cudaDev));
  // CUDACHECK(cudaMallocAsync((void**)&ttt, sendcount * comm->nRanks* ncclTypeSize(datatype), stream));
  // CUDACHECK(cudaMallocAsync((void**)&rrr, sendcount * comm->nRanks* ncclTypeSize(datatype), stream));
  // CUDACHECK(cudaMemsetAsync(rrr,0,sendcount * comm->nRanks* ncclTypeSize(datatype), stream));
  // CUDACHECK(cudaMemcpyAsync(ttt, sendbuff, sendcount * comm->nRanks* ncclTypeSize(datatype), cudaMemcpyDeviceToDevice, stream));
  // NCCLCHECK(ncclAlltoAll((void*)ttt, (void*)rrr, sendcount, datatype, comm, stream));

  // CUDACHECK(cudaMemcpyAsync(recvbuff, rrr, sendcount * comm->nRanks* ncclTypeSize(datatype), cudaMemcpyDeviceToDevice, stream));




  // SDP4Bit alltoallcomp
  // if(sendbuff !=recvbuff&& comm->rank == 0)
  //   {
  //     // const int numGroups = 16;
  //     // const int groupCounts = decompChunkCount / numGroups;
  //     const int groupCounts = 2048;
  //     const int numGroups = (sendcount + groupCounts - 1) / groupCounts;

  //     uint8_t *recvquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(int8_t));
  //     uint8_t *sendquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(int8_t));
  //     float *send = (float*)malloc(sendcount * comm->nRanks * sizeof(float));
  //     float *recv = (float*)malloc(sendcount * comm->nRanks * sizeof(float));
  //     cudaMemcpyAsync(recv, recvbuff, sendcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
  //     cudaMemcpyAsync(send, sendbuff, sendcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
  //     cudaMemcpyAsync(recvquan, recvCompbuff, compSendCount * comm->nRanks * sizeof(int8_t), cudaMemcpyDeviceToHost,stream);
  //     cudaMemcpyAsync(sendquan, sendCompbuff, compSendCount * comm->nRanks * sizeof(int8_t), cudaMemcpyDeviceToHost,stream);
      
  //     cudaStreamSynchronize(stream);
  //     size_t paramsBytes = 2 * numGroups * sizeof(float);
  //     ALIGN_SIZE(paramsBytes, 32);
  //     size_t quanBytes = compSendCount * sizeof(int8_t);
  //     ALIGN_SIZE(quanBytes, 32);
  //     // CUDACHECK(cudaDeviceSynchronize());
  //     for(int r = 0; r < comm->nRanks; r++){
  //       float *sbuff = send + r * sendcount;
  //       float *rbuff = recv + r * sendcount;
  //       float *sendparams = reinterpret_cast<float *> (sendquan + r * compSendCount);
  //       float *recvparams = reinterpret_cast<float *> (recvquan + r * compSendCount);
  //       uint8_t *squanbuff = sendquan + r * compSendCount + paramsBytes;
  //       uint8_t *rquanbuff = recvquan + r * compSendCount + paramsBytes;
  //         for(int i=0;i<numGroups;i++){            
  //           for(int j=0;j<groupCounts;j++){
  //             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d groupid %d org %f sendquan %d dequan %f recvquan %d  ", 
  //               comm->rank, r, i * groupCounts + j, i, sbuff[i*groupCounts + j], squanbuff[i*groupCounts + j], rbuff[i*groupCounts + j], rquanbuff[i * groupCounts + j]);
  //             // printf("ori: %f, quan: %u", host_input[r*groupCounts+i], quan[r*groupCounts+i]);
  //           }
  //         }
  //         for(int i=0;i<2 * numGroups;i++){
  //           INFO(NCCL_INIT,"sendparams: %f recvparams: %f",sendparams[i],recvparams[i]);
  //         }
  //     }
  //     free(recvquan);
  //     free(sendquan);
  //     free(send);
  //     free(recv);
  //   }



  // SDP4Bit alltoallcomp
  // if(sendbuff !=recvbuff&& comm->rank == 0)
  //   {
  //     // const int numGroups = 16;
  //     // const int groupCounts = decompChunkCount / numGroups;
  //     const int groupCounts = 32;
  //     const int numGroups = (sendcount + groupCounts - 1) / groupCounts;

  //     uint8_t *recvquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(int8_t));
  //     uint8_t *sendquan = (uint8_t*)malloc(compSendCount * comm->nRanks * sizeof(int8_t));
  //     __nv_bfloat16 *send = (__nv_bfloat16*)malloc(sendcount * comm->nRanks * sizeof(__nv_bfloat16));
  //     __nv_bfloat16 *recv = (__nv_bfloat16*)malloc(sendcount * comm->nRanks * sizeof(__nv_bfloat16));
  //     cudaMemcpyAsync(recv, recvbuff, sendcount * comm->nRanks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,stream);
  //     cudaMemcpyAsync(send, sendbuff, sendcount * comm->nRanks * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost,stream);
  //     // cudaMemcpyAsync(recvquan, recvCompbuff, compSendCount * comm->nRanks * sizeof(int8_t), cudaMemcpyDeviceToHost,stream);
  //     // cudaMemcpyAsync(sendquan, sendCompbuff, compSendCount * comm->nRanks * sizeof(int8_t), cudaMemcpyDeviceToHost,stream);
      
  //     cudaStreamSynchronize(stream);
  //     size_t paramsBytes = 1 * numGroups * sizeof(__nv_bfloat16);
  //     ALIGN_SIZE(paramsBytes, 32);
  //     size_t quanBytes = compSendCount * sizeof(int8_t);
  //     ALIGN_SIZE(quanBytes, 32);
  //     // CUDACHECK(cudaDeviceSynchronize());
  //     for(int r = 0; r < comm->nRanks; r++){
  //       __nv_bfloat16 *sbuff = send + r * sendcount;
  //       __nv_bfloat16 *rbuff = recv + r * sendcount;
  //       // uint8_t *squanbuff = sendquan + r * compSendCount + paramsBytes;
  //       // uint8_t *rquanbuff = recvquan + r * compSendCount + paramsBytes;
  //         for(int i=0;i<numGroups;i++){            
  //           for(int j=0;j<groupCounts;j++){
  //             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d groupid %d org %f sendquan %d dequan %f recvquan %d  ", 
  //             //   comm->rank, r, i * groupCounts + j, i, sbuff[i*groupCounts + j], squanbuff[i*groupCounts + j], rbuff[i*groupCounts + j], rquanbuff[i * groupCounts + j]);
  //             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d groupid %d org %f dequan %f", 
  //                 comm->rank, r, i * groupCounts + j, i, __bfloat162float(sbuff[i*groupCounts + j]), __bfloat162float(rbuff[i*groupCounts + j]));           
  //           }
  //         }
  //         // for(int i=0;i<2 * numGroups;i++){
  //         //   INFO(NCCL_INIT,"sendparams: %f recvparams: %f",sendparams[i],recvparams[i]);
  //         // }
  //     }
  //     free(recvquan);
  //     free(sendquan);
  //     free(send);
  //     free(recv);
  //   }



// allreduce
// if(sendbuff !=recvbuff && comm->rank == 0)
// {
//   float *send = (float*)malloc(count * sizeof(float));
//   float *recv = (float*)malloc(count * sizeof(float));
//   // uint8_t *recvT = (uint8_t*)malloc(compSendCount*comm->nRanks * ncclTypeSize(compDatatype));
//   cudaMemcpyAsync(recv, recvbuff, count * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   // cudaMemcpyAsync(recvT, recvCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(send, sendbuff, count * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaStreamSynchronize(stream);

//   // CUDACHECK(cudaDeviceSynchronize());
//   for(int r=0;r<comm->nRanks;r++){
          
//         for(int i=0;i<chunkCount;i++){
//             // const float min_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount));
//             // const float max_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount + sizeof(float)));
//             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f reduced %f",comm->rank, r ,i ,send[r*chunkCount+i], recv[r*chunkCount+i]);
//             // else
//             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f",comm->rank, r ,i , send[r*recvcount+i], recvT[r*recvcount+i]);
//         }

//         INFO(NCCL_INIT, "chunkid %d: ", r);
//   }
//   INFO(NCCL_INIT,"rank: %d", comm->rank);
//   // free(quan);
//   free(send);
//   // free(recvT);
//   free(recv);
// }



// // reduce scatter comp
// if(sendbuff !=recvbuff && comm->rank == 0)
//   {

//   float *send = (float*)malloc(recvcount*comm->nRanks * sizeof(float));
//   float *recv = (float*)malloc(recvcount* sizeof(float));
//   cudaMemcpyAsync(recv, recvbuff, recvcount * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaMemcpyAsync(send, sendbuff, recvcount * comm->nRanks * sizeof(float), cudaMemcpyDeviceToHost,stream);
//   cudaStreamSynchronize(stream);

//   // CUDACHECK(cudaDeviceSynchronize());
//    for(int r=0;r<comm->nRanks;r++){
        
//         for(int i=0;i<recvcount;i++){
//             if(r==comm->rank){
//             // const float min_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount));
//             // const float max_ = *(reinterpret_cast<float *>(recvCompbuff + r * recvcount + sizeof(float)));
//             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f min %f max %f tmprecv %u reduced %f",comm->rank, r ,i ,send[r*recvcount+i], *(reinterpret_cast<float *>(recvT + r * compSendCount)), 
//             // *(reinterpret_cast<float *>(recvT + r * compSendCount + sizeof(float))), recvT[r*compSendCount+32+i], recv[i]);
//             INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f reduced %f",comm->rank, r ,i ,send[r*recvcount+i], recv[i]);
//           }
//             // else
//             // INFO(NCCL_INIT, "rank %d chunkid %d elemIdx %d org %f tmprecv %f",comm->rank, r ,i , send[r*recvcount+i], recvT[r*recvcount+i]);
//         }

//         INFO(NCCL_INIT, "chunkid %d: ", r);
//   }
//   INFO(NCCL_INIT,"rank: %d", comm->rank);
//   // free(quan);
//   free(send);
//   free(recv);
//   }
// reuse buff may have some wrong, some data may not send/recv sometimes
    // NCCLCHECK(ncclCompress(sendbuff, &sendCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    //                     ncclCommOp_t::ReduceScatter, stream));

    //   if(rsCompBuffHandle == nullptr){
    //     void* tempCompbuff = nullptr;
    //     NCCLCHECK(ncclCompress(sendbuff, &tempCompbuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    //       ncclCommOp_t::ReduceScatter, stream));
    //     CUDACHECK(cudaStreamSynchronize(stream));

    //     rSCompBuffSize = compSendCount * comm->nRanks * ncclTypeSize(compDatatype) * 2;
    //     NCCLCHECK(ncclMemAlloc(&rSCompBuff, rSCompBuffSize));
    //     NCCLCHECK(ncclCommRegister(comm, rSCompBuff, rSCompBuffSize, &rsCompBuffHandle));
    //     CUDACHECK(cudaMemcpy(rSCompBuff, tempCompbuff, compSendCount * comm->nRanks * ncclTypeSize(compDatatype), cudaMemcpyDeviceToDevice));
    //     CUDACHECK(cudaDeviceSynchronize());
    //     CUDACHECK(cudaFreeAsync(tempCompbuff, stream));
    //   }
    // else{
    //   NCCLCHECK(ncclCompress(sendbuff, &rSCompBuff, recvcount, datatype, &compSendCount, &compDatatype, comm->nRanks, 
    //     ncclCommOp_t::ReduceScatter, stream));
    // }