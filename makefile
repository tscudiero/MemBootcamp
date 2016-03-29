
BASIC_OPTS = -lineinfo -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -Xcompiler=-fopenmp 
PERF_OPTS = -O3 
DEBUG_OPTS = -g -G 
CACHE_OPTS = -Xptxas="-dlcm=ca"

all:
	nvcc $(BASIC_OPTS) $(PERF_OPTS) $(CACHE_OPTS) main.cu -o bootcamp

debug:
	nvcc $(BASIC_OPTS) $(DEBUG_OPTS) $(CACHE_OPTS) main.cu  -o bootcamp

clean:
	rm -f bootcamp

