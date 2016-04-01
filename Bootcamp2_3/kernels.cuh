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

#ifndef __BOOTCAMP_KERNELS_H__
#define __BOOTCAMP_KERNELS_H__

template <typename T>
__global__ void reference_algo (long n,
								long m,
								long * offsetArray, 
								T * dataArray,
								int * iterationArray, // Outer Loop Limit
								int summands,         // Inner Loop Limit
								 T * results)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int iterations =  iterationArray[idx];
	if (idx >= m)
	{ return;}

	long offset = offsetArray[idx];
	for(int i=0; i<iterations; i++)
	{
		for(int s=0; s<summands; s++)
		{
			results[idx] += dataArray[offset + s];
		}
		offset = offsetArray[offset];
	}
}

template<typename T>
__global__ void random_access_pattern(long n,
									  long m,
									  long * offsetArray, 
									  T * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  T * results)
{	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	long offset = idx;
	if(idx >= m)
		{ return; }

	int iterations = LDG(&iterationArray[idx]);
	offset = LDG(&offsetArray[offset]);
	for(int i=0; i<iterations; i++)
	{
		T r = 0.0;
		#pragma unroll 8
		for(int s=0; s<summands && (offset + s) < n; s++)
		{
			r += LDG(&dataArray[offset+s]);
		}
		offset = LDG(&offsetArray[offset]);
		results[idx] += r;
	}
}

template<typename data_t, typename loadAs_t, int SIZE_RATIO>
__global__ void random_access_loadAs (long n, 
									  long m,
									  long * offsetArray, 
									  data_t * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  data_t * results)
{	
	union ACCESSOR
	{
		loadAs_t loadValue;
		data_t   readValues[SIZE_RATIO];
	};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= m)
		{return;}

	int iterations =  LDG(&iterationArray[idx]);
	data_t r = 0.0;
	ACCESSOR v;
	long offset = idx;
	//It must be the case that dataArray[0] is aligned on the boundary
	// of the larger type.
 	ASSERT((unsigned long) dataArray % sizeof(loadAs_t) == 0);
	// For simplified bounds checking, we assume that all summands fit nicely
	//  into an integer number of ACCESSORs
	ASSERT(summands % SIZE_RATIO == 0);

	offset = LDG(&offsetArray[offset]);
	for(int i=0; i<iterations; i++)
	{
		r = 0.0;
		#pragma unroll 4
		for(int s=0; s<summands; s+=SIZE_RATIO)
		{
			v.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s]));
			#pragma unroll 4
			for(int k=0; k< SIZE_RATIO; k++)
			{
				r+= v.readValues[k];
			}
		}
		offset = LDG(&offsetArray[offset]);
		results[idx] += r;
	}
}




template<typename data_t, typename loadAs_t, int SIZE_RATIO>
__global__ void random_access_loadAs_s2 (long n,
										 long m,
									  	 long * offsetArray, 
									  	 data_t * dataArray,
									  	 int *  iterationArray, // Outer Loop Limit
									  	 int summands,   // Inner Loop Limit
									  	 data_t * results)
{	
	union ACCESSOR
	{
		loadAs_t loadValue;
		data_t   readValues[SIZE_RATIO];
	};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= m)
		{return;}
	int iterations =  LDG(&iterationArray[idx]);
	data_t r = 0.0;
	ACCESSOR v0, v1;
	long offset = idx;
	//It must be the case that dataArray[0] is aligned on the boundary
	// of the larger type.
 	ASSERT((unsigned long) dataArray % sizeof(loadAs_t) == 0);
	// For simplified bounds checking, we assume that all summands fit nicely
	//  into an integer number of ACCESSORs
	ASSERT(summands % SIZE_RATIO == 0);

	offset = LDG(&offsetArray[offset]);
	for(int i=0; i<iterations; i++)
	{
		r =0.0;
		for(int s=0; s<summands; s+=SIZE_RATIO*2)
		{
			v0.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s]));
			v1.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s + SIZE_RATIO]));
			for(int k=0; k< SIZE_RATIO; k++)
			{
				r+= v0.readValues[k] + v1.readValues[k];
			}
		}
		offset = LDG(&offsetArray[offset]);
		results[idx] += r;
	}
}

template<typename T>
__global__ void random_access_unrolled_iteration2(long n,
									  long m,
									  long * offsetArray, 
									  T * dataArray,
									  int * iterationArray, // Outer Loop Limit
									  int summands,   // Inner Loop Limit
									  T * results)
{	
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int iterations =  LDG(&iterationArray[idx]);
	T r1 = 0.0;
	T r2 = 0.0;
	long o1 = LDG(&offsetArray[idx]);
	long o2 = LDG(&offsetArray[o1]);

	if(idx < m)
	{
		// Simply assert that we have an even number of 
		// iterations, otherwise more involved bounds checking 
		// is necessary inside teh summands loop.
		ASSERT(iterations%2 == 0);
		#pragma unroll 4
		for(int i=0; i<iterations; i+=2)
		{
			#pragma unroll 16
			for(int s=0; s<summands; s++)
			{
				T a,b;
				a = LDG(&dataArray[o1+s]);
				b = LDG(&dataArray[o2+s]);
				r1 +=a;
				r2 +=b;
			}
			o1 = LDG(&offsetArray[o2]);
			o2 = LDG(&offsetArray[o1]);
		}
		results[idx] += (r1 + r2);
	}
}

template<typename data_t, typename loadAs_t, int SIZE_RATIO>
__global__ void random_access_loadAs_s4 (long n,
										 long m,
									  	 long * offsetArray, 
									  	 data_t * dataArray,
									  	 int *  iterationArray, // Outer Loop Limit
									  	 int summands,   // Inner Loop Limit
									  	 data_t * results)
{	
	union ACCESSOR
	{
		loadAs_t loadValue;
		data_t   readValues[SIZE_RATIO];
	};

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx >= m)
		{return;}
	int iterations =  LDG(&iterationArray[idx]);
	data_t r = 0.0;
	ACCESSOR v0, v1, v2, v3;
	long offset = idx;
	//It must be the case that dataArray[0] is aligned on the boundary
	// of the larger type.
 	ASSERT((unsigned long) dataArray % sizeof(loadAs_t) == 0);
	// For simplified bounds checking, we assume that all summands fit nicely
	//  into an integer number of ACCESSORs
	ASSERT(summands % SIZE_RATIO == 0);

	offset = LDG(&offsetArray[offset]);
	for(int i=0; i<iterations; i++)
	{
		r =0.0;
		for(int s=0; s<summands; s+=SIZE_RATIO*4)
		{
			v0.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s]));
			v1.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s + SIZE_RATIO]));
			v2.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s + 2*SIZE_RATIO]));
			v3.loadValue = LDG(reinterpret_cast<loadAs_t*>(&dataArray[offset + s + 3*SIZE_RATIO]));
			for(int k=0; k< SIZE_RATIO; k++)
			{
				r+= v0.readValues[k] + v1.readValues[k] + v2.readValues[k] + v3.readValues[k];
			}
		}
		offset = LDG(&offsetArray[offset]);
		results[idx] += r;
	}
}

#endif