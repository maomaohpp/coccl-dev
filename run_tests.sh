source env.sh
bash build.sh
export NCCL_DEBUG=INFO

export NCCL_P2P_DISABLE=0
# export NCCL_SHM_DISABLE=1
export NCCL_P2P_DIRECT_DISABLE=0
export NCCL_DEBUG_FILE=ncclcomp.%h
export NCCL_ENABLE_COMPRESS=1
export NCCL_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLTOALL_COMPRESS=1
export NCCL_ALLTOALL_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLREDUCE_COMPRESS=1
export NCCL_ALLREDUCE_COMPRESSORS=sdp4bit
export NCCL_ENABLE_ALLGATHER_COMPRESS=1
export NCCL_ALLGATHER_COMPRESSORS=sdp4bit
export NCCL_ENABLE_REDUCESCATTER_COMPRESS=1
export NCCL_REDUCESCATTER_COMPRESSORS=sdp4bit
export NCCL_REDUCESCATTER_INTER_COMPRESSORS=sdp4bit
export NCCL_COMPRESSORS_CONFIG_PATH=/home/konghr/SDP4Bit-COCCL/coccl/src/device/compress/configs
export NCCL_COMPRESSORS_LIB_PATH=/home/konghr/SDP4Bit-COCCL/coccl/build/obj/device/compress/libcompress
# export NCCL_LOCAL_REGISTER=0
export NCCL_CHECKS_DISABLE=1
export NCCL_LOCAL_REGISTER=1

export NCCL_DEBUG=INFO



# echo '------------------------------------------------------alltoall_float_comp_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_comp_perf -d float -b 64K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------alltoall_float_native_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_perf -d float -b 64K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1


# echo '------------------------------------------------------alltoall_half_comp_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_comp_perf -d half -b 32K -e 4G  -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------alltoall_half_native_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_perf -d half -b 32K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------alltoall_bfloat16_comp_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_comp_perf -d bfloat16 -b 32K -e 4G  -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------alltoall_bfloat16_native_perf------------------------------------------------------'
# ../coccl-tests/build/alltoall_perf -d bfloat16 -b 32K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_comp_perf_float------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_perf -d float -b 64K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_comp_twoshot_perf_float------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_twoshot_perf -d float -b 64K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_native_perf_float------------------------------------------------------'
# ../coccl-tests/build/all_gather_perf -d float -b 64K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1


# echo '------------------------------------------------------all_gather_comp_perf_half------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_perf -d half -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_comp_twoshot_perf_half------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_twoshot_perf -d half -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_native_perf_half------------------------------------------------------'
# ../coccl-tests/build/all_gather_perf -d half -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_comp_perf_bfloat16------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_perf -d bfloat16 -b 32K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_comp_twoshot_perf_bfloat16------------------------------------------------------'
# ../coccl-tests/build/all_gather_comp_perf -d bfloat16 -b 32K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_gather_native_perf_bfloat16------------------------------------------------------'
# ../coccl-tests/build/all_gather_perf -d bfloat16 -b 32K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1


echo '------------------------------------------------------reduce_scatter_comp_ring_perf------------------------------------------------------'
/home/konghr/liuxc/nccl-comp-newbuff-tests/build/reduce_scatter_comp_perf -b 16K -e 4G -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------reduce_scatter_comp_oneshot_perf------------------------------------------------------'
# /home/konghr/liuxc/nccl-comp-newbuff-tests/build/reduce_scatter_comp_oneshot_perf -b 16K -e 4G -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------reduce_scatter_comp_twoshot_perf------------------------------------------------------'
# ../coccl-tests/build/reduce_scatter_comp_twoshot_perf -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------reduceScatter_native_perf------------------------------------------------------'
# nsys profile --force-overwrite true -o rs_8Bit ../nccl-comp-newbuff-tests/build/reduce_scatter_perf -b 477M -e 477M -f 2 -t 8 -g 1 -w 50 -n 100 -c 1


# echo '------------------------------------------------------all_reduce_ring_perf------------------------------------------------------'
# ../nccl-comp-newbuff-tests/build/all_reduce_comp_ring_perf -b 16K -e 4G -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_reduce_twoShot_comp_perf------------------------------------------------------'
#  ../nccl-comp-newbuff-tests/build/all_reduce_comp_twoShotall_perf -b 16K -e 4G -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_reduce_twoShot_perf------------------------------------------------------'
# ../nccl-comp-newbuff-tests/build/all_reduce_twoShotall_perf -b 16K -e 4G -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_reduce_comp_oneShot_perf------------------------------------------------------'
# ../nccl-comp-newbuff-tests/build/all_reduce_comp_oneShot_perf -b 16K -e 128M -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_reduce_oneShot_perf------------------------------------------------------'
# ../nccl-comp-newbuff-tests/build/all_reduce_oneShot_perf -b 16K -e 128M -f 2 -t 8 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------all_reduce_native_perf------------------------------------------------------'
# ../coccl-tests/build/all_reduce_perf -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1

# echo '------------------------------------------------------reduceScatter_native_perf------------------------------------------------------'

# ../coccl-tests/build/reduce_scatter_perf -b 16K -e 4G -f 2 -t 4 -g 1 -w 50 -n 100 -c 1
