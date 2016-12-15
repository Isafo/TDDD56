/*
 * Image filter in OpenCL
 */

#define IS_LOCAL 0
#define KERNELSIZE 4
#define BLOCK_SIZE 16

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int width, const unsigned int height)
{ 

  	int k, l;
  	unsigned int sumx, sumy, sumz;
  	unsigned int i = get_global_id(1) % 512;
  	unsigned int j = get_global_id(0) % 512;

#if IS_LOCAL == 1

	int localSize = BLOCK_SIZE+2*(KERNELSIZE-1);
	__local unsigned char localmem[BLOCK_SIZE + KERNELSIZE*2][BLOCK_SIZE + KERNELSIZE*2][3];

  	unsigned int ii = get_local_id(1);
  	unsigned int jj = get_local_id(0);
    
	if (j < width && i < height) // If inside image
	{
		// this pixel data
		int imgInd = (i * width + j) * 3;
		localmem[ii + KERNELSIZE][jj + KERNELSIZE][0] = image[(i * width + j) * 3 + 0];
		localmem[ii + KERNELSIZE][jj + KERNELSIZE][1] = image[(i * width + j) * 3 + 1];
		localmem[ii + KERNELSIZE][jj + KERNELSIZE][2] = image[(i * width + j) * 3 + 2];

		if (i > KERNELSIZE && i < height - KERNELSIZE && j > KERNELSIZE && j < width - KERNELSIZE)
		{
			// read 2 pixels up padding
			if(ii < KERNELSIZE)
			{
				int imgInd = ((i - ii - 1) * width + j) * 3;
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE][0] = image[imgInd + 0];
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE][1] = image[imgInd + 1];
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE][2] = image[imgInd + 2];
			}

			// bottom pixel
			if(ii > BLOCK_SIZE - KERNELSIZE - 1)
			{
				int imgInd = ((i + KERNELSIZE) * width + j) * 3;
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE][0] = image[imgInd + 0];
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE][1] = image[imgInd + 1];
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE][2] = image[imgInd + 2];
			}

			// lefts padding
			if(jj < KERNELSIZE)
			{
				int imgInd = (i * width + j - jj - 1) * 3;
				localmem[ii + KERNELSIZE][KERNELSIZE - jj - 1][0] = image[imgInd + 0];
				localmem[ii + KERNELSIZE][KERNELSIZE - jj - 1][1] = image[imgInd + 1];
				localmem[ii + KERNELSIZE][KERNELSIZE - jj - 1][2] = image[imgInd + 2];
			}

			// right padding
			if(jj > BLOCK_SIZE - KERNELSIZE - 1)
			{
				int imgInd = (i * width + j + KERNELSIZE) * 3;
				localmem[ii + KERNELSIZE][jj + KERNELSIZE * 2][0] = image[imgInd + 0];
				localmem[ii + KERNELSIZE][jj + KERNELSIZE * 2][1] = image[imgInd + 1];
				localmem[ii + KERNELSIZE][jj + KERNELSIZE * 2][2] = image[imgInd + 2];
			}

			// corner
			if(ii < KERNELSIZE && jj < KERNELSIZE)
			{
				int imgInd = ((i - ii - 1) * width + j - jj - 1) * 3;
				localmem[KERNELSIZE - ii - 1][KERNELSIZE - jj - 1][0] = image[imgInd + 0];
				localmem[KERNELSIZE - ii - 1][KERNELSIZE - jj - 1][1] = image[imgInd + 1];
				localmem[KERNELSIZE - ii - 1][KERNELSIZE - jj - 1][2] = image[imgInd + 2];
			}
			
			if(ii > BLOCK_SIZE - KERNELSIZE - 1 && jj < KERNELSIZE)
			{
				int imgInd = ((i + KERNELSIZE) * width + j - jj - 1) * 3;
				localmem[ii + KERNELSIZE * 2][KERNELSIZE - jj - 1][0] = image[imgInd + 0];
				localmem[ii + KERNELSIZE * 2][KERNELSIZE - jj - 1][1] = image[imgInd + 1];
				localmem[ii + KERNELSIZE * 2][KERNELSIZE - jj - 1][2] = image[imgInd + 2];
			}

			if(ii > BLOCK_SIZE - KERNELSIZE - 1 && jj > BLOCK_SIZE - KERNELSIZE - 1)
			{
				int imgInd = ((i + KERNELSIZE) * width + j + KERNELSIZE) * 3;
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE * 2][0] = image[imgInd + 0];
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE * 2][0] = image[imgInd + 1];
				localmem[ii + KERNELSIZE * 2][jj + KERNELSIZE * 2][0] = image[imgInd + 2];
			}

			if(ii < KERNELSIZE && jj > BLOCK_SIZE - KERNELSIZE - 1)
			{
				int imgInd = ((i - ii - 1) * width + j + KERNELSIZE) * 3;
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE * 2][0] = image[imgInd + 0];
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE * 2][0] = image[imgInd + 1];
				localmem[KERNELSIZE - ii - 1][jj + KERNELSIZE * 2][0] = image[imgInd + 2];
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
					sumx += localmem[ii + KERNELSIZE + k][jj + KERNELSIZE + l][0];
					sumy += localmem[ii + KERNELSIZE + k][jj + KERNELSIZE + l][1];
					sumz += localmem[ii + KERNELSIZE + k][jj + KERNELSIZE + l][2];
#endif					
				}
			out[(i*width+j)*3+0] = sumx/divby;
			out[(i*width+j)*3+1] = sumy/divby;
			out[(i*width+j)*3+2] = sumz/divby;			
		}
		else
		// Edge pixels are not filtered
		{
#if IS_LOCAL == 0			
			out[(i*width+j)*3+0] = image[(i*width+j)*3+0];
			out[(i*width+j)*3+1] = image[(i*width+j)*3+1];
			out[(i*width+j)*3+2] = image[(i*width+j)*3+2];
#else
			out[(i*width+j)*3+0] = localmem[ii + KERNELSIZE][jj + KERNELSIZE][0];
			out[(i*width+j)*3+1] = localmem[ii + KERNELSIZE][jj + KERNELSIZE][1];
			out[(i*width+j)*3+2] = localmem[ii + KERNELSIZE][jj + KERNELSIZE][2];
#endif
			
		}
	}
#if IS_LOCAL 0		
	barrier(CLK_LOCAL_MEM_FENCE)
#endif	
}
