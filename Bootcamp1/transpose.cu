#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h>

#include <nvToolsExtCuda.h>

#include "cudaMacros.h"

// Set to 1 to force check the answer each time.
#define VERIFY 0
// Average bandwidth over nReps executions
const int nReps = 40;


template <typename T>
void referenceTranspose(int rows, int cols, T * in, T * out)
{
  #pragma omp parallel for
  for(int i=0; i<rows; i++)
  {
   #pragma omp parallel for
   for(int j=0; j<cols; j++)
    {
      out[i*rows + j] = in[j*cols + i];
    }
  }
}

template <typename T>
__global__ void gpuTranspose1_kernel(int rows, int cols, T * in, T * out)
{
  int i; int j;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  for(j=0; j<cols; j++)
  {
    out[i*rows + j] = in[j*cols + i];
  }
}

template<typename T>
void gpuTranspose1(int rows, int cols, T * in, T * out) 
{
  int threads = 32;
  gpuTranspose1_kernel<T><<<rows/threads, threads>>>(rows,cols,in,out);
  CUDA_CHECK();
}

template <typename T>
__global__ void gpuTranspose2_kernel(int rows, int cols, T * in, T * out)
{
  int i; int j;
  
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  out[i*rows + j] = in[j*cols + i];
}

template <typename T>
void gpuTranspose2(int rows, int cols, T * in, T * out) 
{
  gpuTranspose2_kernel<T><<<dim3(rows/32,cols/32),dim3(32,32)>>>(rows,cols,in,out);
  CUDA_CHECK();
}

template <typename T, int TILE_DIM>
__global__ void gpuTranspose3_kernel(int rows, int cols, T * in, T * out)
{
  int i; int j;
  __shared__ T tile[TILE_DIM][TILE_DIM];

  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  // Stage Matrix tile to Shared memory
  tile[threadIdx.y][threadIdx.x] = in[j*cols + i];

  __syncthreads();

  i = blockIdx.y * blockDim.y + threadIdx.x;
  j = blockIdx.x * blockDim.x + threadIdx.y;
  // Write matrix tile back out, transposed
  out[j*rows + i] = tile[threadIdx.x][threadIdx.y];
}

template <typename T, int TILE_DIM>
void gpuTranspose3(int rows, int cols, T * in, T * out) 
{
  gpuTranspose3_kernel<T, TILE_DIM><<<dim3(rows/TILE_DIM,cols/TILE_DIM),dim3(TILE_DIM,TILE_DIM)>>>(rows,cols,in,out);
  CUDA_CHECK();
}



template <typename T, int TILE_DIM>
__global__ void gpuTranspose4_kernel(int rows, int cols, T * in, T * out)
{
  int i; int j;
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  // Stage Matrix tile to Shared memory
  tile[threadIdx.y][threadIdx.x] = in[j*cols + i];

  __syncthreads();

  i = blockIdx.y * blockDim.y + threadIdx.x;
  j = blockIdx.x * blockDim.x + threadIdx.y;
  // Write matrix tile back out, transposed
  out[j*rows + i] = tile[threadIdx.x][threadIdx.y];
}
template <typename T, int TILE_DIM>
void gpuTranspose4(int rows, int cols, T * in, T * out) {
  gpuTranspose4_kernel<T, TILE_DIM><<<dim3(rows/TILE_DIM,cols/TILE_DIM),dim3(TILE_DIM, TILE_DIM)>>>(rows,cols,in,out);
  CUDA_CHECK();
}

template <typename T, int TILE_DIM, int ELEMENTS_PER_THREAD>
__global__ void gpuTranspose5_kernel(int rows, int cols, T * in, T * out)
{
  int i, j;
  __shared__ T tile[TILE_DIM][TILE_DIM+1];
  
  int BLOCKY = (TILE_DIM/ELEMENTS_PER_THREAD);
  i=blockDim.x*blockIdx.x+threadIdx.x;
  j=blockDim.y*blockIdx.y*ELEMENTS_PER_THREAD+threadIdx.y;
  
  #pragma unroll
  for(int e=0;e<ELEMENTS_PER_THREAD;e++) 
  {
    tile[threadIdx.y+e*BLOCKY][threadIdx.x] = in[(j+e*BLOCKY)*cols + i];
  }
  
  __syncthreads();
  
  i=blockDim.y*blockIdx.y*ELEMENTS_PER_THREAD+threadIdx.x;
  j=blockDim.x*blockIdx.x+threadIdx.y;
  #pragma unroll
  for(int e=0;e<ELEMENTS_PER_THREAD;e++) 
  {
    out[(j+e*BLOCKY)*rows + i] = tile[threadIdx.x][threadIdx.y+e*BLOCKY];
  }
}
template <typename T, int TILE_DIM, int ELEMENTS_PER_THREAD>
void gpuTranspose5(int rows, int cols, T * in, T * out) 
{
  assert(TILE_DIM%ELEMENTS_PER_THREAD == 0);
  gpuTranspose5_kernel<T,TILE_DIM, ELEMENTS_PER_THREAD><<<dim3(rows/TILE_DIM,cols/TILE_DIM),dim3(TILE_DIM,TILE_DIM/ELEMENTS_PER_THREAD)>>>(rows,cols,in,out);
  CUDA_CHECK();
}

template <typename T>
void time_kernel(char* label, void (*fptr)(int, int, T*, T*),  int rows, int cols, T * in, T * out) 
{
  double start, end;

  start = omp_get_wtime();
  for(int i=0; i<nReps; i++)
  {
    fptr(rows, cols, in, out);
  }
  end = omp_get_wtime();
 
  if(VERIFY)
  {
    nvtxRangePush("Verification");
    // Check the answer: return time only if answer is correct. This section part is not timed.
    T * validation = (T*)malloc(rows*cols*sizeof(T));
    referenceTranspose<T>(rows, cols, in, validation);
    for(long i=0; i<rows*cols; i++)
    {
      if(out[i] != validation[i])
      {
        printf("Error: transpose result is incorrect at linear index %ld\n", i);
        return;
      }
    }
    free(validation); 
    nvtxRangePop();
  }

  double kTime = (end - start)/((double) nReps);
  double kBandwidth = 2.0 * (double)(rows*cols*sizeof(T)) / (1000*1000*1000*kTime);
  printf("%s: Kernel bandwidth: %f gb/sec\n", label, kBandwidth);

}

template <typename T>
void time_kernel_cuda(char* label, void (*fptr)(int, int, T*, T*),  int rows, int cols, T * in, T * out) 
{
  double start, end;

  T * devA;
  T * devAtrans;
  CUDA_CALL(cudaMalloc((void**)&devA, rows*cols*sizeof(T))); 
  CUDA_CALL(cudaMalloc((void**)&devAtrans, rows*cols*sizeof(T))); 
  CUDA_CALL(cudaMemcpy(devA, in, rows*cols*sizeof(T), cudaMemcpyHostToDevice)); 

  CUDA_CALL(cudaDeviceSynchronize()); 
  start = omp_get_wtime();
  for(int i=0; i<nReps; i++)
  {
    fptr(rows, cols, devA, devAtrans); 
  }
  CUDA_CALL(cudaDeviceSynchronize()); 
  end = omp_get_wtime();

  CUDA_CALL(cudaMemcpy(out, devAtrans, rows*cols*sizeof(T), cudaMemcpyDeviceToHost)); 

  if(VERIFY)
  {
    nvtxRangePush("Verification");
    // Check the answer: return time only if answer is correct. This section part is not timed.
    T * validation = (T*)malloc(rows*cols*sizeof(T));
    referenceTranspose<T>(rows, cols, in, validation);
    for(long i=0; i<rows*cols; i++)
    {
      if(out[i] != validation[i])
      {
        printf("Error: transpose result is incorrect at linear index %ld\n", i);
        return;
      }
    }
    free(validation);
    nvtxRangePop();
  }
  CUDA_CALL(cudaFree(devA));
  CUDA_CALL(cudaFree(devAtrans));

  double kTime = (end - start)/((double) nReps);
  double kBandwidth = 2.0 * (double)(rows*cols*sizeof(T)) / (1000*1000*1000*kTime);
  printf("%s: Kernel bandwidth: %f gb/sec\n", label, kBandwidth);
}


template <typename T, int TILE_DIM>
void transposeDriver(int M, int N)
{
  T * A = NULL;
  T * Atrans = NULL;
  A = (T*)malloc(M*N*sizeof(T));
  Atrans = (T*)malloc(M*N*sizeof(T));

  printf("OpenMP Threads: %d\n",omp_get_max_threads());

  // Initialize A to random values and zero Atrans
  for(int i=0; i<M*N; i++)
  {
    A[i] = rand() / static_cast<T>(RAND_MAX);
  }
  memset(Atrans, 0, M*N*sizeof(T));
  if(sizeof(T) == 8)
  {
      printf("Setting 8 byte bank access\n"); 
     CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
   }
  else
     {printf("Setting 4 byte bank access\n"); CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));}
  
  time_kernel<T>((char*)"CPU+OMP", referenceTranspose<T>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-1D", gpuTranspose1<T>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-2D", gpuTranspose2<T>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-shared", gpuTranspose3<T, TILE_DIM>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-no-conflicts", gpuTranspose4<T, TILE_DIM>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-multi-element  2", gpuTranspose5<T, TILE_DIM, 2>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-multi-element  4", gpuTranspose5<T, TILE_DIM, 4>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-multi-element  8", gpuTranspose5<T, TILE_DIM, 8>, M, N, A, Atrans);
  time_kernel_cuda<T>((char*)"CUDA-multi-element 16", gpuTranspose5<T, TILE_DIM, 16>, M, N, A, Atrans);


  free(A);
  free(Atrans);

}


int main(int argc, char** argv)
{
  static int M = 2*1024;
  static int N = 2*1024;
    transposeDriver<float, 32>(M,N);
    transposeDriver<double, 32>(M,N);
  return 0;
}
