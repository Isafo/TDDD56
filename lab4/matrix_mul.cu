
// Question answers:
//
// 1 ) int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = indx + indy * blockDim.x;

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__
void mat_mul(float* a, float* b, float* c, float* input) 
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
	
	mat_mul<<<dimGrid, dimBlock>>>(ad, bd, cd, N);
	cudaThreadSynchronize();
	
	// cudaMemCpy(dest, src, datasize, arg)
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			printf("%f ", c[i+j*N]);
		}

	printf("\n");
	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
