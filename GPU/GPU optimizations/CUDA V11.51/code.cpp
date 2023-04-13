//__global__ void matrixMul_tiling_coalescing(float* c, float* a, float* b, int m, int n, int k)
//{
//	// Block index and thread index
//	int by = blockIdx.y;
//	int bx = blockIdx.x;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//	int iy = BLOCK_SIZE * by + ty;
//	int ix = BLOCK_SIZE * bx + tx;
//
//	// shared memory to hold sub-matrix of A and B
//	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
//	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
//
//	// index and numbers
//	int aBegin = k * BLOCK_SIZE * by;
//	int aEnd = aBegin + k - 1;
//	int aStep = BLOCK_SIZE;
//	int bBegin = k * BLOCK_SIZE * bx;
//	int bStep = BLOCK_SIZE;
//
//	// temp sum
//	float Csub = 0;
//	if (ix < n && iy < m)
//	{
//#pragma unroll
//		for (int i = aBegin, j = bBegin;
//			i <= aEnd;
//			i += aStep, j += bStep)
//		{
//			// Load sub-matrix into shared memory
//			As[ty][tx] = a[i + k * ty + tx];
//			Bs[ty][tx] = b[j + k * ty + tx];
//
//			// Synchronize to make sure the matrices are loaded
//			__syncthreads();
//
//			// multiply two sub-matrices
//			// Note that here exists bank conflicts and we should try to avoid it
//			for (int l = 0; l < BLOCK_SIZE; l++)
//				Csub += As[ty][l] * Bs[l][tx];
//
//			// sync to make sure all temp results are correctly computed
//			__syncthreads();
//		}
//		// Write results back to matrix C
//		int cBegin = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
//		c[cBegin + n * ty + tx] = Csub;
//	}
//}