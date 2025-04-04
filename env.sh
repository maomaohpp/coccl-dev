export CUDA_PATH=/usr/local/cuda-12.6
export PATH=$CUDA_PATH/bin:$PATH
export CUDACXX=$CUDA_PATH/bin/nvcc
export NVHPC_CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64/:$LD_LIBRARY_PATH


# export OMPI_DIR=/usr/mpi/gcc/openmpi-4.1.7a1/
# export PATH=$OMPI_DIR/bin:$PATH
# export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7a1/
# export LD_LIBRARY_PATH=$OMPI_DIR/lib:$LD_LIBRARY_PATH


export NCCL_HOME=/home/konghr/SDP4Bit-COCCL/coccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export PATH=$NCCL_HOME/lib:$PATH


# export LD_LIBRARY_PATH=/home/konghr/miniconda3/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=/home/konghr/miniconda3/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/konghr/liuxc/CUPTI/lib64:$LD_LIBRARY_PATH
