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

#ifndef __CUDA_API_APPLICATION_MACROS__
#define __CUDA_API_APPLICATION_MACROS__

#include <assert.h>

#define EXIT_ON_ERROR        0
#define SEGFAULT_ON_ERROR 1024
#define CONTINUE_ON_ERROR 4096

#define ERROR_BEHAVIOR EXIT_ON_ERROR

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
{	printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__);\
    	if(ERROR_BEHAVIOR==EXIT_ON_ERROR) { exit(-1);} \
    	if(ERROR_BEHAVIOR==SEGFAULT_ON_ERROR) {volatile int * x = 0; printf("Segfault is intentional: %d\n", x[0]);}}


#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess )        \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__,__LINE__-1);\
    	if(ERROR_BEHAVIOR==EXIT_ON_ERROR) { exit(-1);} \
    	if(ERROR_BEHAVIOR==SEGFAULT_ON_ERROR) { volatile int * x = 0; printf("Segfault is intentional: %d\n", x[0]);}}

#define USE_LDG 1
#define DISABLE_ASSERTIONS 0

#if USE_LDG
  #define LDG(x) __ldg(x)
#else
  #define LDG(x) (*x)
#endif

#if DISABLE_ASSERTIONS
	#define ASSERT(x)
#else
	#define ASSERT(x) assert(x)
#endif

#endif
