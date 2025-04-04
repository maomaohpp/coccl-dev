source env.sh
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
cd tests/coccl-tests
make -j CUDA_HOME=$NVHPC_CUDA_HOME NCCL_HOME=$NCCL_HOME NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"