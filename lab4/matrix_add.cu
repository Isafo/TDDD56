
// Question answers:
//
// 1 ) int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = indx + indy * blockDim.x;
//
// 2 ) CUDA: N=16 blz=16 0.06
//	     N=32	 0.066
//	     N=64	 0.077
//	     N=128	 0.126
//	     N=256 	 0.276
//	     N=512	 1.341
//	     N=1024	 2.972

// 	CPU: N=16	 0.004
//	     N=32	 0.013
//	     N=64	 0.068
//	     N=128	 0.328
//	     N=256	 2.334

// 3 ) N=16 blz=16 0.06
//    N=16 blz=32 0.05
//    N=16 blz=64 0.04
//    N=16 blz=128 0.04

// 4 ) 0.293 vs 0.276

#include <stdio.h>

const int N = 256; 
const int blocksize = 16; 

__global__
void mat_add(float* a, float* b, float* c) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}

int main()
{
	
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

	float *ad, *bd, *cd;
	const int size = N*N*sizeof(float);

	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );	
	cudaMalloc( (void**)&cd, size );
	
	dim3 dimBlock( blocksize*blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	
	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice ); 
	
	cudaEvent_t e_start;
	cudaEventCreate(&e_start);
	cudaEventRecord(e_start, 0);

	mat_add<<<dimGrid, dimBlock>>>(ad, bd, cd);
	cudaThreadSynchronize();
	
	// cudaMemCpy(dest, src, datasize, arg)
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

	cudaEvent_t e_stop;
	cudaEventCreate(&e_stop);
	cudaEventRecord(e_stop, 0);

	cudaEventSynchronize(e_start);
	cudaEventSynchronize(e_stop);

	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	float time;
	cudaEventElapsedTime(&time, e_start, e_stop);


	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			printf("%f ", c[i+j*N]);
		}

	printf("\n");
	delete[] c;
	
	printf("done, time: %f \n", time);
	return EXIT_SUCCESS;
}
