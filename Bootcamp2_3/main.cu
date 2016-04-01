/*
 *  Copyright 2016 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdio.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <omp.h>
#include <stdlib.h>


#include "cudaMacros.h"
#include "initKernels.cuh"
#include "kernels.cuh"
#include "collaborativeKernels.cuh"

 #define DEBUG_DATA_ARRAYS 0

template <typename T>
inline double gen_checksum(T * dataArray, long m)
{
	double sum = 0;
	for(long i=0; i<m; i++)
	{
		sum += static_cast<double>(dataArray[i]);
	}
	return sum;
}

template<typename T>
double verifyGPU(T * devResults, long m)
{
	T * verificationArray;
	CUDA_CALL(cudaMallocHost(&verificationArray, m*sizeof(T)));
	CUDA_CALL(cudaMemcpy(verificationArray, devResults, m*sizeof(T), cudaMemcpyDeviceToHost));
	double ret = gen_checksum<T>(verificationArray, m);
	CUDA_CALL(cudaFreeHost(verificationArray));
	return ret;
}

long sumIterations(int * hostIterationArray,long m)
{
	long r = 0;
	for(long i=0; i<m; i++)
	{
		r+= hostIterationArray[i];
	}
	return r;

}
long countIterations(int * devIterationArray, long m)
{
	int * hostIterationArray;
	CUDA_CALL(cudaMallocHost(&hostIterationArray, m*sizeof(int)));
	CUDA_CALL(cudaMemcpy(hostIterationArray, devIterationArray, m*sizeof(int), cudaMemcpyDeviceToHost));
	long r = sumIterations(hostIterationArray, m);
	CUDA_CALL(cudaFreeHost(hostIterationArray));
	return r;
}

template <typename T>
void cpu_reference_algo (long n,
	                     long m, 
						 long * offsetArray, 
						 T * dataArray,
						 int * iterationArray, // Outer Loop Limit
						 int summands,         // Inner Loop Limit
						 T * results)
{

	#pragma omp parallel for
	for(long idx=0; idx <m; idx++)
	{
		int iterations =  iterationArray[idx];
		long offset = offsetArray[idx];
		T L = 0;
		for(int i=0; i<iterations; i++)
		{
			for(int s=0; s<summands; s++)
			{
				L += dataArray[offset + s];
			}
			offset = offsetArray[offset];
		}
		results[idx] = L;
	}
}


template <typename T, int unroller>
void cpu_optimized_algo (long n,
				         long m,
						 long * offsetArray, 
						 T * dataArray,
						 int * iterationArray, // Outer Loop Limit
						 int summands,   // Inner Loop Limit
						 T * results)
{

	#pragma omp parallel for
	for(long idx=0; idx <m; idx++)
	{
		int iterations =  iterationArray[idx];
		long offset = offsetArray[idx];
		T L = 0;
		T loader[unroller];
		for(int i=0; i<iterations; i++)
		{
			for(int s=0; s<summands; s+= unroller)
			{
				for(int u=0; u<unroller; u++)
				{
					loader[u] = dataArray[offset + s + u];	
				}

				for(int u=0; u<unroller; u++)
				{
					L += loader[u];
				}

			}
			offset = offsetArray[offset];
		}
		results[idx] = L;
	}
}

template <typename T>
void cpu_inner_driver(std::string name, 
						void (*f)(long, long, long*, T*, int*,int, T*), 
	                    long n, 
	                    long m, 
	                    long * offsetArray, 
	                    T * dataArray, 
	                    int * iterationArray,
	                    int summands, 
	                    T* resultArray)
{
	printf("%s:\n", name.c_str());
	memset(resultArray, 0, m*sizeof(T));
	double start = omp_get_wtime();
	f(n, m, offsetArray, dataArray, iterationArray, summands, resultArray);
	double end = omp_get_wtime();
	long totalIterations = sumIterations(iterationArray, m);
	double duration = end-start;
	
	
	double checksum = gen_checksum<T>(resultArray, m);

	double bytesMoved =  static_cast<double>(sizeof(T)) *(
						(static_cast<double>(totalIterations) * static_cast<double>(summands)) + // algo reads
						(static_cast<double>(m))        // write result at the end (this is probably in the noise)
						);
	double bandwidth = bytesMoved/ (1.0E9 * duration);
	int numThreads= -1;
	char * ompEnvVar;
	ompEnvVar = getenv("OMP_NUM_THREADS");
	if(ompEnvVar != NULL)
	{
		numThreads = atoi(ompEnvVar);
	}
	printf("\t(CPU (%d threads): %f(%f GB/s), \t\t %ld\n", numThreads, duration, bandwidth, static_cast<long>(checksum));

}

template <typename T>
void cpu_driver(long n,
	            long m, 
				long * devOffsetArray, 
				T * devDataArray,
				int * devIterationArray, // Outer Loop Limit
				int summands,   // Inner Loop Limit
				T * devResultArray)
{
	T * hostDataArray;
	T * hostResultArray;
	int * hostIterationArray;
	long * hostOffsetArray;
	CUDA_CALL(cudaMallocHost(&hostOffsetArray, n*sizeof(long)));
	CUDA_CALL(cudaMallocHost(&hostIterationArray, m*sizeof(int)));
	CUDA_CALL(cudaMallocHost(&hostDataArray, n*sizeof(T)));
	CUDA_CALL(cudaMallocHost(&hostResultArray, m*sizeof(T)));

	CUDA_CALL(cudaMemcpy(hostOffsetArray, devOffsetArray, n*sizeof(long), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(hostIterationArray, devIterationArray, m*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(hostDataArray, devDataArray, n*sizeof(T), cudaMemcpyDeviceToHost));

	cpu_inner_driver<T>(std::string("CPU Reference Algo"), cpu_reference_algo<T>, n, m, hostOffsetArray, hostDataArray, hostIterationArray, summands, hostResultArray);
	cpu_inner_driver<T>(std::string("CPU Unroller 2"), cpu_optimized_algo<T,2>, n, m, hostOffsetArray, hostDataArray, hostIterationArray, summands, hostResultArray);
	cpu_inner_driver<T>(std::string("CPU Unroller 4"), cpu_optimized_algo<T,4>, n, m,  hostOffsetArray, hostDataArray, hostIterationArray, summands, hostResultArray);

	CUDA_CALL(cudaFreeHost(hostOffsetArray));
	CUDA_CALL(cudaFreeHost(hostIterationArray));
	CUDA_CALL(cudaFreeHost(hostDataArray));
	CUDA_CALL(cudaFreeHost(hostResultArray));
}

// Simple, first principles approach to determining the alignment of a pointer.
//  Note: This is implemented this way for clarity. Bithacking methods will be
//        an order of magniuted faster if this is called repeatedly. This is simply
//        for debugging code in any event.
int findAlignment(void * ptr)
{
	for(int i=1; i<1024*1024; i*=2)
	{
		if( ((unsigned long)ptr) % i !=0 )
		{
			return i/2;
		}
	}
	return 0;
}


template<typename T>
void innerDriver(std::string kernelName, void (*kernel)(long, long, long*, T*, int *, int, T*),
			     long n, long m, long * devOffsetArray, T * devDataArray,
			     int * devIterationArray, int summands, T* devResultArray,
			     int threadLimitLow, int threadLimitHigh)
{
	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));	
	dim3 threadBlock(1,1,1); // Behavior is implicit, but specified explicitly for clarity
	dim3 grid(1,1,1); 
	float kernelTime;
	for(int threads = threadLimitLow; threads <= threadLimitHigh; threads*=2)
	{
		threadBlock.x = threads;
		grid.x = (m+(threadBlock.x-1))/threadBlock.x;
		CUDA_CALL(cudaMemset(devResultArray, 0, m*sizeof(T)));
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaEventRecord(start));
		kernel<<<grid, threadBlock>>>(n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray);
		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));
		CUDA_CALL(cudaEventElapsedTime(&kernelTime, start, stop));
		double checksum = verifyGPU<T>(devResultArray, m);
		long totalIterations = countIterations(devIterationArray, m);
		double bytesMoved =  static_cast<double>(sizeof(T)) *(
						(static_cast<double>(totalIterations) * static_cast<double>(summands)) + // algo reads
						(static_cast<double>(m))        // write result at the end (this is probably in the noise)
						);
		double iterationRate = (totalIterations)/ (1.0E3 * kernelTime);
		double jobRate = m/( kernelTime);
		double bandwidth = bytesMoved/(1.0E6 * kernelTime);
		printf("%s\n\t<<<%d, %d>>> %fms (%f GB/s) (%f M iterations/sec) (%f K jobs/sec), \t %ld\n", kernelName.c_str(), threadBlock.x, grid.x, kernelTime, bandwidth, iterationRate, jobRate, static_cast<long>(checksum));				

	}
}

template<typename T>
void innerDriverVector(std::string kernelName, void (*kernel)(long, long, long*, T*, int *, int, T*),
			     long n, long m, long * devOffsetArray, T * devDataArray,
			     int * devIterationArray, int summands, T* devResultArray,
			     int threadLimitLow, int threadLimitHigh)
{
	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));	
	dim3 threadBlock(1,1,1); // Behavior is implicit, but specified explicitly for clarity
	dim3 grid(1,1,1); 
	float kernelTime;
	for(int threads = threadLimitLow; threads <= threadLimitHigh; threads*=2)
	{
		threadBlock.x = threads;
		// Vector launches one threadblock per task. This difference here is the only place
		// innerDriverVector differs from innerDriver
		grid.x = m;
		CUDA_CALL(cudaMemset(devResultArray, 0, m*sizeof(T)));
		CUDA_CALL(cudaDeviceSynchronize());
		CUDA_CALL(cudaEventRecord(start));
		kernel<<<grid, threadBlock>>>(n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray);
		CUDA_CALL(cudaEventRecord(stop));
		CUDA_CALL(cudaEventSynchronize(stop));
		CUDA_CALL(cudaEventElapsedTime(&kernelTime, start, stop));
		double checksum = verifyGPU<T>(devResultArray, m);
		long totalIterations = countIterations(devIterationArray, m);
		double bytesMoved =  static_cast<double>(sizeof(T)) *(
						(static_cast<double>(totalIterations) * static_cast<double>(summands)) + // algo reads
						(static_cast<double>(m))        // write result at the end (this is probably in the noise)
						);
		double iterationRate = (totalIterations)/ (1.0E3 * kernelTime);
		double jobRate = m/( kernelTime);
		double bandwidth = bytesMoved/(1.0E6 * kernelTime);
		printf("%s\n\t<<<%d, %d>>> %f ms (%f GB/s) (%f M iterations/sec) (%f K jobs/sec), \t %ld\n", kernelName.c_str(), threadBlock.x, grid.x, kernelTime, bandwidth, iterationRate, jobRate, static_cast<long>(checksum));				
	}
}

template<typename T>
void outerDriver(long n, long m, int iterations, int summands, int divergenceFactor=1)
{
	assert( m <= n);
	dim3 threadBlock(32);
	int gridFactor = 16;
	dim3 initGrid((n+(gridFactor*threadBlock.x-1))/(gridFactor*threadBlock.x));
	T * devDataArray;
	long * devOffsetArray;
	int *  devIterationArray;
	T * devResultArray;
	curandState_t * devRandState;
	
	long cuRandStateBytes = initGrid.x*threadBlock.x * sizeof(curandState_t);
	long bytes = n * (sizeof(T) +  sizeof(long)) + m*(sizeof(T) + sizeof(int)) + cuRandStateBytes;
	printf("Data Allocations: %f mb \n", static_cast<double>(bytes)/1.0E6);
	
	CUDA_CALL(cudaMalloc(&devRandState, cuRandStateBytes));
	CUDA_CALL(cudaMalloc(&devOffsetArray, n * sizeof(long)));
	CUDA_CALL(cudaMalloc(&devDataArray, n*sizeof(T)));
	CUDA_CALL(cudaMalloc(&devResultArray, m*sizeof(T)));
	CUDA_CALL(cudaMalloc(&devIterationArray, m*sizeof(int)));

	// This is the maximum element alignment required. E.g. to do loadAs with float4 instead of 4,
	// alignment must be at 4*sizeof(float). 16 bytes is the largest single load command supported
	// by the gpu currently.
	int elementAlignment = 16/sizeof(T);

	init_offsets<<<initGrid, threadBlock>>>(devRandState, n, devOffsetArray, n-summands, elementAlignment);
	int minIterations, maxIterations;
	if(divergenceFactor == 1)
	{
		minIterations = maxIterations = iterations;
	}
	else
	{
		minIterations = static_cast<int>(iterations / divergenceFactor);
		maxIterations = static_cast<int>(iterations);
	}
	init_iterations<<<initGrid, threadBlock>>>(devRandState, m, devIterationArray, minIterations, maxIterations);
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaFree(devRandState));


#if DEBUG_DATA_ARRAYS
	// Debugging: Alignment Verification Block
	{
		printf("Alignment of devDataArray is %d (%p)\n", findAlignment(devDataArray), devDataArray);;
		long * offsetVerification;
		CUDA_CALL(cudaMallocHost(&offsetVerification, n*sizeof(long)));
		CUDA_CALL(cudaMemcpy(offsetVerification, devOffsetArray, n*sizeof(long), cudaMemcpyDeviceToHost));
		for(long v=0; v<n; v++)
		{
			if(offsetVerification[v]%elementAlignment >0)
			{
				printf("Offset %ld of %ld is invalid: %ld mod %d == %ld\n", v, n, offsetVerification[v], elementAlignment, offsetVerification[v]%elementAlignment);
				exit(-1);
			}
		}
		CUDA_CALL(cudaFreeHost(offsetVerification));
	}

	// Debugging: Iteration Verification Block
	{
		printf("Checking stats on iteration array\n");
		int * iterationVerification;
		CUDA_CALL(cudaMallocHost(&iterationVerification, m*sizeof(int)));
		CUDA_CALL(cudaMemcpy(iterationVerification, devIterationArray, m*sizeof(int), cudaMemcpyDeviceToHost));
		double sum = 0.0;
		int max = 0.0;
		int min = 1024*1024;
		for(int i=0; i<m; i++)
		{	
			sum += static_cast<double>(iterationVerification[i]);
			if(iterationVerification[i] > max)
				{ max = iterationVerification[i];}
			if(iterationVerification[i] < min)
				{ min = iterationVerification[i];}
		}
		double mean = sum / static_cast<double>(m);
		double stdev = 0.0;
		for(int i=0; i<m; i++)
		{
			double t = static_cast<double>(iterationVerification[i]) - mean;
			stdev += t*t;
		}
		stdev /= static_cast<double>(m);
		stdev = sqrt(stdev);
		printf("Iteration Verification: [%d, %d], mean=%f, std=%f\n", min, max, mean, stdev);

		for(int i=0; i<32; i++)
		{
			printf("%d  ", iterationVerification[i]);
		}
		printf("\n");
		cudaFreeHost(&iterationVerification);

	}
#endif

	data_init<T><<<initGrid, threadBlock>>>(n, devDataArray);
	CUDA_CALL(cudaDeviceSynchronize());

	CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	innerDriver(std::string("Reference     "), reference_algo<T>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	innerDriver(std::string("Generic     "), random_access_pattern<T>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	

	// Collaborative load kernels
	if( m%32 == 0 && summands%32 == 0)
	{
		if(sizeof(T) == 8)
			{CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));}
		else
			{CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));}
		CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
		innerDriver(std::string("Collaborative: 4"), collaborative_access_small_summandVersatile<T, 32, 4>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Collaborative: 8"), collaborative_access_small_summandVersatile<T, 32, 8>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Collaborative: 16"), collaborative_access_small_summandVersatile<T, 32, 16>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Collaborative: 32"), collaborative_access_small_summandVersatile<T, 32, 32>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	}

	// Vector-style collaborative kernels		
	if(sizeof(T) == 8)
		{CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));}
	else
		{CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));}
	CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	innerDriverVector(std::string("VectorStyle: "), collaborative_access_vectorStyle<T, 32>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	innerDriverVector(std::string("VectorStyle_vecReduce: "), collaborative_access_vectorStyle_vecReduce<T, 32>,  n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);


	CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	if(summands >= 4 && summands%4 == 0 && sizeof(T)==4)
	{
	CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
		innerDriver(std::string("Load As< , float4,>"), random_access_loadAs<T, float4, 4>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Load As s2< , float4,>"), random_access_loadAs_s2<T, float4, 4>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Load As s4< , float4,>"), random_access_loadAs_s4<T, float4, 4>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	}
	if(summands >=2 && summands%2 == 0 && sizeof(T) == 8)
	{
		CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
		innerDriver(std::string("Load As< , double2,>"), random_access_loadAs<T, double2, 2>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Load As s2< , double2,>"), random_access_loadAs_s2<T, double2, 2>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
		innerDriver(std::string("Load As s4< , double2,>"), random_access_loadAs_s4<T, double2, 2>, n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray, 32, 32);
	}

	cpu_driver(n, m, devOffsetArray, devDataArray, devIterationArray, summands, devResultArray);

	CUDA_CALL(cudaFree(devIterationArray));
	CUDA_CALL(cudaFree(devOffsetArray));
	CUDA_CALL(cudaFree(devDataArray));
	CUDA_CALL(cudaFree(devResultArray));
}

void processArgv(int argc, char ** argv, long * n, long * m, int * iterations, int * summands, int * divergence)
{
	int i=1;
	(*n) = 10*1024*1024;
	(*m) = 384*3*1024;
	(*iterations) = 32;
	(*summands) = 32;
	(*divergence) = 1;

	while(i<argc)
	{
		if(0 == strcmp(argv[i], "-h") || 0 == strcmp(argv[i], "-H"))
		{
			printf("Bootcamp 2:\n  Usage: ./bootcamp [option] [value] ... [option] [value]\n");
			printf("  Options:\n");
			printf("\t -h       :  Display this help message and exit\n");
			printf("\t -m Value :  Value is the number of parallel tasks executing: Default %ld\n", (*m));
			printf("\t -n Value :  Value is the size, in millions of elements, of the data table. Default: %ld\n", (*n)/(1024*1024));
			printf("\t -s Value :  Value is the number of summands processed at each memory jump. Default %d\n", (*summands));
			printf("\t -i Value :  Value is the number of iterations(memory jumps) per thread. Default %d\n", (*iterations));
			printf("\t -d Value :  Value is the divergence factor. Default %d\n", (*divergence));
			printf("\n When the divergence factor is not 1, the number of iterations for each task is varies between \n");
			printf("(iterations/divergence factor) and iterations. For example, using -i 64 and -d 2, the number of iterations\n");
			printf("executed by each task will be between 32 and 64. The distribution is uniform across that range.\n");
			printf("\n");
			exit(0);
		}
		
		if(0 == strncmp(argv[i], "-m", 4) || 0 == strncmp(argv[i], "-M", 4))
		{
			(*m) = static_cast<long>(atoi(argv[i+1]));
			i++;
		}
		if(0 == strncmp(argv[i], "-n", 4) || 0 == strncmp(argv[i], "-n", 4))
		{
			(*n) = atoi(argv[i+1])*1024*1024;
			i++;
		}
		if(0 == strncmp(argv[i], "-i", 4) || 0 == strncmp(argv[i], "-I", 4))
		{
			(*iterations) = atoi(argv[i+1]);
			i++;
		}
		if(0 == strncmp(argv[i], "-s", 4) || 0 == strncmp(argv[i], "-S", 4))
		{
			(*summands) = atoi(argv[i+1]);
			i++;
		}
		if(0 == strncmp(argv[i], "-d", 4) || 0 == strncmp(argv[i], "-D", 4))
		{
			(*divergence) = atoi(argv[i+1]);
			i++;
		}
		i++;
	}
}

int main(int argc, char** argv)
{
	
	long n,m; 
	int iterations, summands, divergence;
	processArgv(argc, argv, &n, &m, &iterations, &summands, &divergence);
	printf("===============================\n");
	printf("Bootcamp Configuration:\n");
	printf("===============================\n");
	printf("   N          : %ld elements\n", n);
	printf("   M          : %ld threads\n", m);
	printf("   summands   : %d values\n", summands);
	printf("   iterations : %d times \n", iterations);
	printf("   divergence : %d x\n", divergence);
	printf("===============================\n");

	printf("Running experiment for type Float\n");
	outerDriver<float>(n, m, iterations, summands, divergence);

	printf("Running experiment for type Double\n");
	outerDriver<double>(n, m, iterations, summands, divergence);

	return 0;
}