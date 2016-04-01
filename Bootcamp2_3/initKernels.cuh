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

#ifndef __BOOTCAMP_INIT_KERNELS__
#define __BOOTCAMP_INIT_KERNELS__

/*
 * Init_offsets generates randomized offsets.
 *  Arguments: 
 *		state - a block of uninitialized memory, sizeof(cudrandState_t) * blockDim.x*gridDim.x  bytes in size
 *              in other words, one curandState_t value per thread in the grid that launched this kernel.
 *              The grid which launches this need not be related to the grids used elsewhere.
 *		n  - the number of longs in offsetArray
 * 		offsetArray - the output. At the end, this will be initialized to n offset values.
 *      magnitude - values in offset array will be in the range [0, magnitude-1] at the end of the kernel.
 *      alignment - values in offset array will all be divisible by alignment. Use this to assure that offsets
 *                  in the offset array adhere to some specified alignment. Note however that alignment should
 *                  be specified in array elements, as this routine generates offsets, not addresses.
 * 
 * Note: Since init offsets initializes curand with the threadIdx, if the same sized grid and parameters are used
 *        in more than one call to this routine, the resulting offset arrays will be identical. This behavior 
 *        is intentional.
 */
__global__ void init_offsets(curandState_t * state, long n, long * offsetArray, long magnitude, int alignment)
{
	long idx = threadIdx.x + blockDim.x * blockIdx.x;
	// Initialize with index for repeatability
	curand_init(idx, 0, 0, &state[idx]);
	for(long i=idx; i<n; i+= blockDim.x*gridDim.x)
	{
	
		double f = curand_uniform(&state[idx]);
		double m = static_cast<double>((magnitude-1)/alignment); // Cast to double for precision considerations
		offsetArray[i] = alignment * static_cast<long> ( m*f );
	}
}

/*
 * Init_iterations generates a uniformly distributed randomized number of iterations required by each task between
 *   a lower and upper bound.
 *
 *  Arguments: 
 *		state - a block of uninitialized memory, sizeof(cudrandState_t) * blockDim.x*gridDim.x  bytes in size
 *              in other words, one curandState_t value per thread in the grid that launched this kernel.
 *              The grid which launches this need not be related to the grids used elsewhere.
 *      n - the number of results which will be generated (i.e. number of tasks)
 *		iteratiosArray  - an array of n integers where the results will be stored.
 *      minIterations - the minimum number of iterations which can be returned
 *      maxIterations - the maximum number of iterations which can be returned
 */
__global__ void init_iterations(curandState_t * state, long n, int * iterationsArray, int minIterations, int maxIterations)
{
	long idx = threadIdx.x + blockDim.x * blockIdx.x;
	// Initialize with index for repeatability
	curand_init(idx, 0, 0, &state[idx]);
	for(long i=idx; i<n; i+= blockDim.x*gridDim.x)
	{

		double f = curand_uniform(&state[idx]);
		double m = static_cast<double>(maxIterations - minIterations);
		iterationsArray[i] = minIterations + static_cast<int> ( m*f );
	}
}


/* data_init()
 *     Creates the data array in-place on the GPU in parallel.  This is basically random data 
 *       but it is not, of course, even remotely random 
 */
template<typename T>
__global__ void data_init(long dataSize, T * dataArray)
{
	long idx= threadIdx.x + blockDim.x * blockIdx.x;
	for(long i=idx; i<dataSize; i+= blockDim.x * gridDim.x)
	{
		dataArray[i] = static_cast<T>(i%100);
	}
}


#endif