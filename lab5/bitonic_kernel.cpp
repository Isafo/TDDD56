
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

// If you plan on submitting your solution for the Parallel Sorting Contest,
// please keep the split into main file and kernel file, so we can easily
// insert other data.

__device__ static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__global__ void bitonic_sort_gpu(int *data, int N)
{
	extern __shared__ int cached[];

	const int tid = threadIdx.x;

	cached[tid] = data[tid];

	__syncthreads();

	int i, j, k;
	for (k = 2; k <= N; k = 2 * k) // Outer loop, double size for each step
	{
		for (j = k >> 1; j>0; j = j >> 1) // Inner loop, half size for each step
		{
			int ixj = tid^j; // Calculate indexing!
			if ((ixj) > tid)
			{
				if ((tid & k) == 0 && cached[i] > cached[ixj]) exchange(cached[i], cached[ixj]);
				if ((tid & k) != 0 && cached[i] < cached[ixj]) exchange(cached[i], cached[ixj]);
			}
			__syncthreads();
		}
	}

	data[tid] = cached[tid];
}


// No, this is not GPU code yet but just a copy of the CPU code, but this
// is where I want to see your GPU code!
void bitonic_gpu(int *data, int N)
{
	int* d_data;
	cudaMalloc((void*)&d_data, sizeof(int) * N);
	cudaMemcpy(d_data, data, sizeof(int) * N, cudaMemcpyHostToDevice);

	dim3 gridDim(1, 1);
	dim3 blockSize(N, 1);
	bitonic_sort_gpu <<<gridDim, blockSize, sizeof(int) * N >>>(d_data, N);

	cudaMemcpy(data, d_data, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(d_data);
}
