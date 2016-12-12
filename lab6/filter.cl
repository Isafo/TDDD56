/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int width, const unsigned int height)
{ 
  	unsigned int i = get_global_id(1) % 512;
  	unsigned int j = get_global_id(0) % 512;
  	unsigned int ii = get_local_id(1) % 512;
  	unsigned int jj = get_local_id(0) % 512;

  	int k, l;
  	unsigned int sumx, sumy, sumz;
  	__local unsigned char[width*height*3] shared;

  	shared[ii*width + jj] = image[i*width + j];
  	barrier(CLK_LOCAL_MEM_FENCE)

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
					//sumx += image[((i+k)*width+(j+l))*3+0];
					//sumy += image[((i+k)*width+(j+l))*3+1];
					//sumz += image[((i+k)*width+(j+l))*3+2];
					sumx += shared[((i+k)*width+(j+l))*3+0];
					sumy += shared[((i+k)*width+(j+l))*3+1];
					sumz += shared[((i+k)*width+(j+l))*3+2];
				}
			out[(i*width+j)*3+0] = sumx/divby;
			out[(i*width+j)*3+1] = sumy/divby;
			out[(i*width+j)*3+2] = sumz/divby;
		}
		else
		// Edge pixels are not filtered
		{
			out[(i*width+j)*3+0] = shared[(i*width+j)*3+0];
			out[(i*width+j)*3+1] = shared[(i*width+j)*3+1];
			out[(i*width+j)*3+2] = shared[(i*width+j)*3+2];
			
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
