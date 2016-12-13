/*
 * Image filter in OpenCL
 */

#define IS_LOCAL 0
#define KERNELSIZE 2
#define BLOCK_SIZE 5

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int width, const unsigned int height)
{ 

  	int k, l;
  	unsigned int sumx, sumy, sumz;
  	unsigned int i = get_global_id(1) % 512;
  	unsigned int j = get_global_id(0) % 512;

#if IS_LOCAL == 1

	int localSize = BLOCK_SIZE+2*(KERNELSIZE-1);
	__local unsigned char[2*localSize*3] localmen;

  	unsigned int ii = get_local_id(0);
  	unsigned int jj = get_local_id(1);
    
  	// https://www.codeproject.com/KB/showcase/Work-Groups-Sync/image001.gif
  	// get_local_id, get_global_id ...

	if (j < width && i < height) // If inside image
	{
		if (i >= KERNELSIZE && i < height-KERNELSIZE && j >= KERNELSIZE && j < width-KERNELSIZE)
		{
			// -/-j
			if(i < KERNELSIZE && j < KERNELSIZE)
			{
				  	localmen[ii*localSize+jj)*3+0] = image[((i-KERNELSIZE)*width+(j-KERNELSIZE))*3+0]
					localmen[ii*localSize+jj)*3+1] = image[((i-KERNELSIZE)*width+(j-KERNELSIZE))*3+1]
					localmen[ii*localSize+jj)*3+2] = image[((i-KERNELSIZE)*width+(j-KERNELSIZE))*3+2]
			}

			// +i/-j
			if(i > KERNELSIZE && j < KERNELSIZE)
			{
				  	localmen[ii*localSize+jj)*3+0] = image[((i+KERNELSIZE)*width+(j-KERNELSIZE))*3+0]
					localmen[ii*localSize+jj)*3+1] = image[((i+KERNELSIZE)*width+(j-KERNELSIZE))*3+1]
					localmen[ii*localSize+jj)*3+2] = image[((i+KERNELSIZE)*width+(j-KERNELSIZE))*3+2]
			}

			// -i/+j
			if(i < KERNELSIZE && j > KERNELSIZE)
			{
				  	localmen[ii*localSize+jj)*3+0] = image[((i-KERNELSIZE)*width+(j+KERNELSIZE))*3+0]
					localmen[ii*localSize+jj)*3+1] = image[((i-KERNELSIZE)*width+(j+KERNELSIZE))*3+1]
					localmen[ii*localSize+jj)*3+2] = image[((i-KERNELSIZE)*width+(j+KERNELSIZE))*3+2]
			}

			// +i/+j
			if(i > KERNELSIZE && j > KERNELSIZE)
			{
				  	localmen[ii*localSize+jj)*3+0] = image[((i+KERNELSIZE)*width+(j+KERNELSIZE))*3+0]
					localmen[ii*localSize+jj)*3+1] = image[((i+KERNELSIZE)*width+(j+KERNELSIZE))*3+1]
					localmen[ii*localSize+jj)*3+2] = image[((i+KERNELSIZE)*width+(j+KERNELSIZE))*3+2]		
			}
		}
	}

  	barrier(CLK_LOCAL_MEM_FENCE)

#endif
  	

	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
	
	if (j < width && i < height) // If inside image
	{
		if (i >= KERNELSIZE && i < height-KERNELSIZE && j >= KERNELSIZE && j < width-KERNELSIZE)
		{
		// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
				{
#if IS_LOCAL == 0
					sumx += image[((i+k)*width+(j+l))*3+0];
					sumy += image[((i+k)*width+(j+l))*3+1];
					sumz += image[((i+k)*width+(j+l))*3+2];
#else

					sumx += localmen[((ii+k)*localSize+(jj+l))*3+0]
					sumy += localmen[((ii+k)*localSize+(jj+l))*3+1]
					sumz += localmen[((ii+k)*localSize+(jj+l))*3+2]			
#endif					
				}
			out[(i*width+j)*3+0] = sumx/divby;
			out[(i*width+j)*3+1] = sumy/divby;
			out[(i*width+j)*3+2] = sumz/divby;			
		}
		else
		// Edge pixels are not filtered
		{
#if IS_LOCAL 0			
			out[(i*width+j)*3+0] = image[(i*width+j)*3+0];
			out[(i*width+j)*3+1] = image[(i*width+j)*3+1];
			out[(i*width+j)*3+2] = image[(i*width+j)*3+2];
#else
			out[(i*width+j)*3+0] = localmen[((ii+k)*localSize+(jj+l))*3+2];
			out[(i*width+j)*3+1] = localmen[((ii+k)*localSize+(jj+l))*3+2];
			out[(i*width+j)*3+2] = localmen[((ii+k)*localSize+(jj+l))*3+2];
#endif
			
		}
	}
#if IS_LOCAL 0		
	barrier(CLK_LOCAL_MEM_FENCE)
#endif	
}
