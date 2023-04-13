#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <random>


#define M 2560
#define N 2560
#define K 2560
#define BLOCK_SIZE 16


__global__ void matrixMul_naive(float* c, float* a, float* b, int m, int n, int k)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;

	float temp_c = 0.0;
	
	if (tx < n && ty < m)
	{
		for (int i = 0; i < k; i++)
		{
			temp_c += a[ty*k + i] * b[i*k + tx];
		}
		c[ty*n + tx] = temp_c;
	}
}
__global__ void matrixMul_tiling(float* c, float* a, float* b, int m, int n,int k)
{
	// Block index and thread index
	int by=blockIdx.y;
	int bx=blockIdx.x;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int iy=BLOCK_SIZE*by+ty;
	int ix=BLOCK_SIZE*bx+tx;

	// shared memory to hold sub-matrix of A and B
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

	// index and numbers
	int aBegin = k * BLOCK_SIZE * by;
	int aEnd = aBegin + k - 1;
	int aStep = BLOCK_SIZE;
	int bBegin = BLOCK_SIZE * bx;
	int bStep = BLOCK_SIZE * n;

	// temp sum
	float Csub = 0;
	if(ix < n && iy < m)
	{
		#pragma unroll
		for (int i = aBegin, j = bBegin;
			i <= aEnd;
			i += aStep, j += bStep) 
		{
			// Load sub-matrix into shared memory
			As[ty][tx] = a[i + k * ty + tx];
			Bs[ty][tx] = b[j + n * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// multiply two sub-matrices
			// Note that here exists bank conflicts and we should try to avoid it
			for (int l = 0; l < BLOCK_SIZE; l++)
				Csub += As[ty][l] * Bs[l][tx];

			// sync to make sure all temp results are correctly computed
			__syncthreads();
		}
		// Write results back to matrix C
		int cBegin = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		c[cBegin + n * ty + tx] = Csub;	
	}
}
__global__ void matrixMul_coalescing(float* C, float*A, float*B, int wA, int wC)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = by * blockDim.y + ty;
	int j = bx * blockDim.x + tx;

	float sum = 0.0;
	for (int k = 0; k < wA; k++)
	{
		sum += A[i*wA + k] * B[j*wA + k];
	}
	C[i*wC + j] = sum;
}

void set_random(float* matrix, int size)
{
	for (int i = 0; i < size; i++)
		matrix[i] = rand() / (float)RAND_MAX;
}
void set_zero(float* matrix, int size)
{
	for (int i = 0; i < size; i++)
		matrix[i] = rand() / (float)RAND_MAX;
}
void transpose(float* matrix, int h, int w)
{
	for (int i = 0; i < w; i++)
	{
		float tmp = 0;
		for (int j = 0; j < h; j++)
		{
			tmp = matrix[i *h + j];
			matrix[i*h + j] = matrix[i + j * w];
			matrix[i + j * w] = tmp;
		}
	}

}
void naive_test()
{
	/******************************
		Naive solution
	******************************/

	// preparations 
	float flop = 2 * (float)M*(float)N*(float)K;
	float sec = 0.0;
	cudaEvent_t start;
	cudaEvent_t end;

	//setting seeds for random
	srand(2021);

	//Host memory and initialization
	float* h_A = (float*)malloc(M*K * sizeof(float));
	float* h_B = (float*)malloc(K*N * sizeof(float));
	float* h_C = (float*)malloc(M*N * sizeof(float));

	set_random(h_A, M*K);
	set_random(h_B, K*N);
	set_zero(h_C, M*N);

	//device memory
	float* d_A;
	cudaMalloc((void**)&d_A, M*K * sizeof(float));
	float* d_B;
	cudaMalloc((void**)&d_B, K*N * sizeof(float));
	float* d_C;
	cudaMalloc((void**)&d_C, M*N * sizeof(float));

	//allocate grids and threads
	dim3 threads, grid;
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3(M / threads.x, N / threads.y);

	//start clocking
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	//memory copy Host to device
	cudaMemcpy(d_A, h_A, M*K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K*N * sizeof(float), cudaMemcpyHostToDevice);
	matrixMul_naive << <grid, threads >> > (d_C, d_A, d_B, M, N, K);
	//device to host
	cudaMemcpy(h_C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost);
	//end clocking
	cudaEventCreate(&end);
	cudaEventRecord(end, NULL);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&sec, start, end);


	//result presentation
	printf("Naive GPU\n");
	printf("Processing time: %f ms,GFLOPS: %f \n", sec, flop / sec / 1e+6);

	//free space
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
void tiling_test()
{
	/******************************
		Tiling solution
	******************************/

	// preparations 
	float flop = 2 * (float)M*(float)N*(float)K;
	float sec = 0.0;
	cudaEvent_t start;
	cudaEvent_t end;

	//setting seeds for random
	srand(2021);

	//Host memory and initialization
	float* h_A = (float*)malloc(M*K * sizeof(float));
	float* h_B = (float*)malloc(K*N * sizeof(float));
	float* h_C = (float*)malloc(M*N * sizeof(float));

	set_random(h_A, M*K);
	set_random(h_B, K*N);
	set_zero(h_C, M*N);

	//device memory
	float* d_A;
	cudaMalloc((void**)&d_A, M*K * sizeof(float));
	float* d_B;
	cudaMalloc((void**)&d_B, K*N * sizeof(float));
	float* d_C;
	cudaMalloc((void**)&d_C, M*N * sizeof(float));

	//allocate grids and threads
	dim3 threads, grid;
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3(M / threads.x, N / threads.y);

	//start clocking
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	//memory copy Host to device
	cudaMemcpy(d_A, h_A, M*K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K*N * sizeof(float), cudaMemcpyHostToDevice);
	matrixMul_tiling << <grid, threads >> > (d_C, d_A, d_B, M, N, K);
	//device to host
	cudaMemcpy(h_C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost);

	//end clocking
	cudaEventCreate(&end);
	cudaEventRecord(end, NULL);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&sec, start, end);


	//result presentation
	printf("Tiling GPU\n");
	printf("Processing time: %f ms,GFLOPS: %f \n", sec, flop / sec / 1e+6);

	//free space
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
void coalescing_test()
{
	/******************************
		coalescing
	******************************/

	// preparations 
	float flop = 2 * (float)M*(float)N*(float)K;
	float sec = 0.0;
	cudaEvent_t start;
	cudaEvent_t end;

	//setting seeds for random
	srand(2021);

	//Host memory and initialization
	float* h_A = (float*)malloc(M*K * sizeof(float));
	float* h_B = (float*)malloc(K*N * sizeof(float));
	float* h_C = (float*)malloc(M*N * sizeof(float));

	set_random(h_A, M*K);
	set_random(h_B, K*N);
	transpose(h_B, K, N);
	set_zero(h_C, M*N);

	//device memory
	float* d_A;
	cudaMalloc((void**)&d_A, M*K * sizeof(float));
	float* d_B;
	cudaMalloc((void**)&d_B, K*N * sizeof(float));
	float* d_C;
	cudaMalloc((void**)&d_C, M*N * sizeof(float));

	//allocate grids and threads
	dim3 threads, grid;
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
	grid = dim3(M / threads.x, N / threads.y);

	//start clocking
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	//memory copy Host to device
	cudaMemcpy(d_A, h_A, M*K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, K*N * sizeof(float), cudaMemcpyHostToDevice);
	matrixMul_coalescing << <grid, threads >> > (d_C, d_A, d_B, K, K);
	//device to host
	cudaMemcpy(h_C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost);

	//end clocking
	cudaEventCreate(&end);
	cudaEventRecord(end, NULL);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&sec, start, end);


	//result presentation
	printf("Coalescing GPU\n");
	printf("Processing time: %f ms,GFLOPS: %f \n", sec, flop / sec / 1e+6);

	//free space
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main()
{
	naive_test();
	tiling_test();
	coalescing_test();
	return 0;
}