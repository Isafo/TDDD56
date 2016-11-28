// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

// Question answers:
//
// 1 ) 1 block is 16 threads and the grid contains 1 block -> 16 threads
//     1 block is 1 SM -> 16 SM
//
// 2 ) the square roots calculated using CUDA cannot be assumed to be the same assumed
//     to be the same as if they were caculate on the CPU. The CUDA computations will
//     contain some error.


#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
//void simple(float *c)
void simple(float* c, float* input) 
{
	//c[threadIdx.x] = threadIdx.x ;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] =  sqrtf(input[index]);
}

int main()
{
	float *c = new float[N];
	for(int i = 0; i < N; ++i)
		c[i] = i;

	float *cd, *bd;
	const int size = N*sizeof(float);
	
	cudaMalloc( (void**)&cd, size );
	cudaMalloc( (void**)&bd, size );
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	
	cudaMemcpy( bd, c, size, cudaMemcpyHostToDevice ); 
	
	simple<<<dimGrid, dimBlock>>>(cd, bd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	// cudaMemCpy(dest, src, datasize, arg)

	cudaFree( cd );
	cudaFree( bd );
	
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");
	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
