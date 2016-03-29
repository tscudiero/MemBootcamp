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

#ifndef __BOOTCAMP_COLLABORATIVE_KERNELS_H__
#define __BOOTCAMP_COLLABORATIVE_KERNELS_H__

template<typename T, int MAX_SUMMANDS>
__global__ void collaborative_access_vectorStyle (long n,
	                                  long m,
									  long * offsetArray, 
									  T * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  T * results)
{
	__shared__ T summandBuffer[MAX_SUMMANDS];
	
	int idx = blockIdx.x;
	if (idx >= m)
	{ return;}

	int iterations =  LDG(&iterationArray[idx]);
	long offset = offsetArray[idx];
	
	int summandPasses = (summands + blockDim.x -1) / blockDim.x;
	
	for(int i=0; i<iterations; i++)
	{
		T r = 0;
		for(int s=0; s<summandPasses; s++)
		{
			int summandIndex = s*blockDim.x + threadIdx.x;
			if(summandIndex < summands)
			{
				summandBuffer[threadIdx.x] = dataArray[offset + summandIndex];
			}
			else
			{	
				summandBuffer[threadIdx.x] = 0;
			}
			__syncthreads();
			if(threadIdx.x == 0)
			{
				for(int i=0; i<blockDim.x; i++)
				{
					r+= summandBuffer[i];
				}

			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			results[idx] += r;
		}
		offset = offsetArray[offset];
	}
}

template<typename T, int MAX_SUMMANDS>
__global__ void collaborative_access_vectorStyle_vecReduce(long n,
	                                  long m,
									  long * offsetArray, 
									  T * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  T * results)
{
	__shared__ T summandBuffer[MAX_SUMMANDS];
	
	int idx = blockIdx.x;
	if (idx >= m)
	{ return;}

	int iterations =  LDG(&iterationArray[idx]);
	long offset = offsetArray[idx];
	
	int summandPasses = (summands + blockDim.x -1) / blockDim.x;
	
	for(int i=0; i<iterations; i++)
	{
		for(int s=0; s<summandPasses; s++)
		{
			int summandIndex = s*blockDim.x + threadIdx.x;
			if(summandIndex < summands)
			{
				summandBuffer[threadIdx.x] = dataArray[offset + summandIndex];
			}
			else
			{	
				summandBuffer[threadIdx.x] = 0;
			}
			__syncthreads();
			int limit = blockDim.x/2;
			while(limit > 0)
			{
				if(threadIdx.x < limit)
				{
					summandBuffer[threadIdx.x] += summandBuffer[limit + threadIdx.x];
				}
				limit /= 2;
				__threadfence_block();
			}
			__syncthreads();
		}
		if(threadIdx.x == 0)
		{
			results[idx] += summandBuffer[0];
		}
		offset = offsetArray[offset];
	}
}

#define COLABORATIVE_CTA_SIZE_MAX 32
#define COLLABORATIVE_MAX_SUMMANDS 32

template<typename T, int MAX_SUMMANDS, int OPS_PER_PASS>
__global__ void collaborative_access_small_summandVersatile (long n,
	                                  long m,
									  long * offsetArray, 
									  T * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  T * results)
{	
	__shared__ long iterationOffset[COLABORATIVE_CTA_SIZE_MAX];
	__shared__ T sdata[OPS_PER_PASS][MAX_SUMMANDS+1];
	__shared__ T sResult[COLABORATIVE_CTA_SIZE_MAX];
	__shared__ int sIterCount[COLABORATIVE_CTA_SIZE_MAX];
	__shared__ int maxIt;

	ASSERT(summands <=COLLABORATIVE_MAX_SUMMANDS);

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	sIterCount[threadIdx.x] =  LDG(&iterationArray[idx]);
	long offset = LDG(&offsetArray[idx]);
	sResult[threadIdx.x] = 0;

	if(threadIdx.x == 0)
	{
		int lMax = sIterCount[0];
		for(int t=1; t<blockDim.x; t++)
		{ 
			if(sIterCount[t] > lMax)
			{ lMax = sIterCount[t];}
		}
		maxIt = lMax;
	}
	int summandPasses;
	int passes;
	if(summands > MAX_SUMMANDS)
	{
		summandPasses = summands/MAX_SUMMANDS;
		passes = MAX_SUMMANDS/OPS_PER_PASS;
	}
	else
	{
		passes = summands/OPS_PER_PASS;
		summandPasses = 1;
	}
	// Assert that the number of summands has divided out equally into a number of actual passes which
	// is the product of summand passes and passes with a full OPS_PER_PASS in each pass.
	ASSERT(summandPasses * passes * OPS_PER_PASS == summands);

	for(int i=0; i< maxIt; i++)
	{	
		iterationOffset[threadIdx.x] = offset;
		sResult[threadIdx.x] = 0;

		for(int p=0; p<passes; p++)
		{
			for(int t=0; t<OPS_PER_PASS; t++)
			{
				sdata[t][threadIdx.x] = 0;
				for(int s=0; s<summandPasses; s++)
				{
					long ldTgt = iterationOffset[p*OPS_PER_PASS + t] + s*MAX_SUMMANDS + threadIdx.x;
					if(ldTgt < n && i < sIterCount[p*OPS_PER_PASS + t])
					{
						sdata[t][threadIdx.x] += LDG(&dataArray[ldTgt]);
					}
				}
			}
			__syncthreads();
			// Perform Additions
			T r =0.0;
			if(threadIdx.x < OPS_PER_PASS)
			{
				for(int s=0; s<summands && s < MAX_SUMMANDS; s++)
				{
					r+= sdata[threadIdx.x][s];
				}
				sResult[p*OPS_PER_PASS + threadIdx.x] += r;
			}
			__syncthreads();
		}
		results[idx] += sResult[threadIdx.x];
		offset = LDG(&offsetArray[offset]);
	}
}


#endif
