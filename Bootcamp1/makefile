NVCC_OPTS = -gencode arch=compute_52,code=sm_52 --generate-code arch=compute_35,code=sm_35 -O3 -Xcompiler=-fopenmp -lineinfo 
PTXAS_OPTS = -Xptxas="-dlcm=ca" 
all:
	nvcc $(NVCC_OPTS) $(PTXAS_OPTS) -l nvToolsExt transpose.cu -o bootcamp_transpose
