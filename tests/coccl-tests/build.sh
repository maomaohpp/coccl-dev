source ../nccl-comp/env.sh
make -j CUDA_HOME=$NVHPC_CUDA_HOME NCCL_HOME=$NCCL_HOME NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"