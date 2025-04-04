SHELL := /usr/bin/env bash
MAKEFLAGS += -r
.SUFFIXES:
.SECONDARY:

# Directories
NCCLDIR := ../../../..
include $(NCCLDIR)/makefiles/common.mk
include $(NCCLDIR)/makefiles/version.mk

BUILDDIR ?= $(abspath ../../../../build)
# OBJDIR := $(BUILDDIR)/obj/device/compress/minmaxUint8/

OBJDIR := $(SUBOBJDIR)/minmaxUint8/
# $(info PWD is ${PWD})
# $(info SUBOBJDIR is ${SUBOBJDIR})


# QUAN_SO := $(BUILDDIR)/obj/device/compress/libcompress/libsdp4bit.so
QUAN_SO := $(SUBOBJDIR)/libcompress/libminmaxUint8.so

ifeq ($(NVHPC_CUDA_HOME),)
  NVHPC_CUDA_HOME := /usr/local/cuda
endif

CUDA_LIB := $(NVHPC_CUDA_HOME)/lib64
CUDA_INC := $(NVHPC_CUDA_HOME)/include

# Compiler flags
NVCC_GENCODE ?= -gencode=arch=compute_80,code=sm_80

INCFLAGS := -I$(CUDA_INC) \
            -I./includes \
            -I. -I../.. -I$(BUILDDIR)/include -I../../../include \
# -I. -I$(INCDIR)
# @echo $(INCFLAGS)

NVCUFLAGS += $(INCFLAGS) \
            -Xcompiler -fPIC \
            $(NVCC_GENCODE)
# -std=c++17 
NVLDFLAGS := -L$(CUDA_LIB)


CXXFLAGS += $(INCFLAGS)
# Output files
# EXEC = $(BUILDDIR)/spd4bit_test
# SO_LIB = $(BUILDDIR)/spd4bit_quan.so

# Source files
SRCS = $(wildcard ./*.cu ./*.cc )
$(info OBJDIR is ${OBJDIR})

# OBJS = $(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SRCS:.cc=.o))
OBJS = $(patsubst %, $(OBJDIR)/%.o, $(SRCS))
$(info OBJS is ${OBJS})

# OBJS := $(OBJS:.cu=.o)
DEPS = $(OBJS:.o=.d)

all: $(QUAN_SO)

SAY = @bash -c 'path="$$2"; [[ "$$(realpath "$$2")" =~ ^$(subst .,\.,$(abspath $(NCCLDIR)))/(.*)$$ ]] && path="$${BASH_REMATCH[1]}"; printf "%-15s %s\n" "$$1" "$$path"' SAY

COMPILE.cu = $(NVCC) $(NVCUFLAGS) -dc $2 -o $1
COMPILE.cpp = $(CXX) $(CXXFLAGS) -c $2 -o $1
COMPILE.cc = $(CXX) $(CXXFLAGS) -c $2 -o $1
define COMPILE
@$(SAY) "Compiling" $2;\
 mkdir -p $(dir $1);\
 $(call COMPILE$(suffix $2),$1,$2)
endef


DEPENDS.cu = $(NVCC) $(NVCUFLAGS) -M -dc $1
DEPENDS.cc = $(CXX) $(CXXFLAGS) -M -c $1
DEPENDS.cpp = $(CXX) $(CXXFLAGS) -M -c $1
define DEPENDS
@$(SAY) "Dependencies" $2;\
 mkdir -p $(dir $1);\
 mk=$$($(call DEPENDS$(suffix $2),$2));\
 [[ $$mk =~ ^[^:]*:(.*)$$ ]];\
 files=$${BASH_REMATCH[1]};\
 files=$$(for x in $$files; do case "$$x" in '\'|$$'\t') ;; *) echo "$$x"; esac; done);\
 files=$$(for x in $$files; do [[ "$$(realpath "$$x")" == "$$(realpath "$(NCCLDIR)")"* ]] && echo "$$x"; done);\
 echo "$(patsubst %.d,%.o,$1) $1: " $$files > $1
endef



$(OBJDIR)/%.o: % $(OBJDIR)/%.d
	$(call COMPILE,$@,$<)

$(OBJDIR)/%.d: %
	$(call DEPENDS,$@,$<)

$(QUAN_SO): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCUFLAGS) $(NVLDFLAGS) $(OBJS) -shared  -o $@




# $(EXEC): $(OBJS)
# 	@mkdir -p $(dir $@)
# 	$(NVCC) $(OBJS) $(NVCUFLAGS) $(NVLDFLAGS)  -o $@ 
# $(OBJDIR)/sdp4bit_test: $(LIB_OBJS)
#     $(NVCC) $(NVCUFLAGS) $^ $(NVLDFLAGS) -o $@ \
#             -D_GLIBCXX_USE_CXX11_ABI=0


# $(OBJDIR)/libs: $(LIB_OBJS)
# 	@$(NVCC) $^ $(NVCUFLAGS) $(NVLDFLAGS) \
#            -shared -o $(OBJDIR)/spd4bit_quan.so -D_GLIBCXX_USE_CXX11_ABI=0
#            ar rcs $(OBJDIR)/spd4bit_quan.a $^

# Compile CUDA source to object files
# $(OBJDIR)/%.o: % $(OBJDIR)/%.d
# 	$(call COMPILE,$@,$<)

# # Generate dependencies for CUDA and C++ files
# $(OBJDIR)/%.d: %
# 	$(call DEPENDS,$@,$<)

-include $(wildcard $(OBJDIR)/*.d)


# $(OBJDIR)/genobj/%.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/%.d
# 	$(call COMPILE,$@,$(OBJDIR)/gensrc/$*)


# %.o: %.cu
# 	@$(NVCC) -c $< -o $@ $(COMMON_INCLUDES) $(NVCUFLAGS) \
#             --compiler-options '-fPIC'  \
#             -D_GLIBCXX_USE_CXX11_ABI=0
# #'--expt-relaxed-constexpr'
# # 链接规则
# $(OUTPUT_NAME): $(OBJ_FILES)
# 	@$(NVCC) $^ $(COMMON_INCLUDES) $(COMMON_LDFLAGS) \
#             -o $@ -D_GLIBCXX_USE_CXX11_ABI=0
.PHONY: clean
clean:
	rm -rf $(OBJDIR)
	rm -f $(SO_LIB)
