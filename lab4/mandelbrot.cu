// Question answers:
//
// 1 ) calculating the x and y using the thread and block id and dims
//
// 2 ) 1024 (DIM/16) blocks and 256 threads per block
//
// 3 ) __device__
//
// 4 ) GPU is much faster
//
// 5 ) float: 0.12 double 0.63
//
// 6 ) By using many blocks loadbalancing becomes less of a problem

#include <stdio.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Image data
	unsigned char	*pixels = NULL;
	int	 gImageWidth, gImageHeight;

// Init image data
void initBitmap(int width, int height)
{
	if (pixels) free(pixels);
	pixels = (unsigned char *)malloc(width * height * 4);
	gImageWidth = width;
	gImageHeight = height;
}

#define DIM 512

// Select precision here! float or double!
#define MYFLOAT float

// User controlled parameters
int maxiter = 20;
MYFLOAT offsetx = -200, offsety = 0, zoom = 0;
MYFLOAT scale = 1.5;

// Complex number class
struct cuComplex
{
    MYFLOAT   r;
    MYFLOAT   i;
    
    __device__ cuComplex( MYFLOAT a, MYFLOAT b ) : r(a), i(b)  {}
    
    __device__ float magnitude2( void )
    {
        return r * r + i * i;
    }
    
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    
    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int mandelbrot(int x, int y, MYFLOAT scale, MYFLOAT ox, MYFLOAT oy, MYFLOAT zoom)
{

    MYFLOAT jx = scale * (float)(DIM/2 - x + ox/scale)/(DIM/2);
    MYFLOAT jy = scale * (float)(DIM/2 - y + oy/scale)/(DIM/2);

    cuComplex c(jx, jy);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i < 20; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i;
    }

    return i;
}

__global__ void kernel(unsigned char* pxl, MYFLOAT scale, MYFLOAT ox, MYFLOAT oy, MYFLOAT zoom)
{
    int indx = blockIdx.x * blockDim.x + threadIdx.x;
    int indy = blockIdx.y * blockDim.y + threadIdx.y;
    int imgInd = indx + indy * blockDim.x * gridDim.x;

    int fractalValue = mandelbrot(indx, indy, scale, ox, oy, zoom);

    int r, g, b;

    // Colorize it
    r = 255 * fractalValue/20;
    if (r > 255) 
        r = 255 - r;
    
    g = 255 * fractalValue*4/20;
    if (g > 255) 
        g = 255 - g;
    
    b = 255 * fractalValue*20/20;
    if (b > 255) 
        b = 255 - b;
    
    pxl[imgInd * 4 + 0] = r;
    pxl[imgInd * 4 + 1] = g;
    pxl[imgInd * 4 + 2] = b;
    pxl[imgInd * 4 + 3] = 255; 

    return;
}

char print_help = 0;

// Yuck, GLUT text is old junk that should be avoided... but it will have to do
static void print_str(void *font, const char *string)
{
	int i;

	for (i = 0; string[i]; i++)
		glutBitmapCharacter(font, string[i]);
}

void PrintHelp()
{
	if (print_help)
	{
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-0.5, 639.5, -0.5, 479.5, -1.0, 1.0);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColor4f(0.f, 0.f, 0.5f, 0.5f);
		glRecti(40, 40, 600, 440);

		glColor3f(1.f, 1.f, 1.f);
		glRasterPos2i(300, 420);
		print_str(GLUT_BITMAP_HELVETICA_18, "Help");

		glRasterPos2i(60, 390);
		print_str(GLUT_BITMAP_HELVETICA_18, "h - Toggle Help");
		glRasterPos2i(60, 300);
		print_str(GLUT_BITMAP_HELVETICA_18, "Left click + drag - move picture");
		glRasterPos2i(60, 270);
		print_str(GLUT_BITMAP_HELVETICA_18,
		    "Right click + drag up/down - unzoom/zoom");
		glRasterPos2i(60, 240);
		print_str(GLUT_BITMAP_HELVETICA_18, "+ - Increase max. iterations by 32");
		glRasterPos2i(60, 210);
		print_str(GLUT_BITMAP_HELVETICA_18, "- - Decrease max. iterations by 32");
		glRasterPos2i(0, 0);

		glDisable(GL_BLEND);
		
		glPopMatrix();
	}
}

// Compute fractal and display image
void Draw()
{
	unsigned char *d_pxl;
	const int size = DIM * DIM * 4 * sizeof(unsigned char);

	cudaMalloc( (void**)&d_pxl, size );

	dim3 dimGrid( DIM/16, DIM/16);
	dim3 dimBlock( 16, 16 );

	cudaEvent_t e_start;
	cudaEventCreate(&e_start);
	cudaEventRecord(e_start, 0);

	kernel<<<dimGrid, dimBlock>>>(d_pxl, scale, offsetx, offsety, zoom);
	cudaThreadSynchronize();

	cudaEvent_t e_stop;
	cudaEventCreate(&e_stop);
	cudaEventRecord(e_stop, 0);

	cudaEventSynchronize(e_start);
	cudaEventSynchronize(e_stop);

	// cudaMemCpy(dest, src, datasize, arg)
	cudaMemcpy( pixels, d_pxl, size, cudaMemcpyDeviceToHost ); 

	cudaFree( d_pxl );

	float time;
	cudaEventElapsedTime(&time, e_start, e_stop);

	cudaEventDestroy(e_start);
	cudaEventDestroy(e_stop);



// Dump the whole picture onto the screen. (Old-style OpenGL but without lots of geometry that doesn't matter so much.)
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glDrawPixels( gImageWidth, gImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels );
	
	if (print_help)
		PrintHelp();
	
	glutSwapBuffers();
}

char explore = 1;

static void Reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	glLoadIdentity();
	glOrtho(-0.5f, width - 0.5f, -0.5f, height - 0.5f, -1.f, 1.f);
	initBitmap(width, height);

	glutPostRedisplay();
}

int mouse_x, mouse_y, mouse_btn;

// Mouse down
static void mouse_button(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		// Record start position
		mouse_x = x;
		mouse_y = y;
		mouse_btn = button;
	}
}

// Drag mouse
static void mouse_motion(int x, int y)
{
	if (mouse_btn == 0)
	// Ordinary mouse button - move
	{
		offsetx += (x - mouse_x)*scale;
		mouse_x = x;
		offsety += (mouse_y - y)*scale;
		mouse_y = y;
		
		glutPostRedisplay();
	}
	else
	// Alt mouse button - scale
	{
		scale *= pow(1.1, y - mouse_y);
		mouse_y = y;
		glutPostRedisplay();
	}
}

void KeyboardProc(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27: /* Escape key */
	case 'q':
	case 'Q':
		exit(0);
		break;
	case '+':
		maxiter += maxiter < 1024 - 32 ? 32 : 0;
		break;
	case '-':
		maxiter -= maxiter > 0 + 32 ? 32 : 0;
		break;
	case 'h':
		print_help = !print_help;
		break;
	}
	glutPostRedisplay();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( DIM, DIM );
	glutCreateWindow("Mandelbrot explorer (CPU)");
	glutDisplayFunc(Draw);
	glutMouseFunc(mouse_button);
	glutMotionFunc(mouse_motion);
	glutKeyboardFunc(KeyboardProc);
	glutReshapeFunc(Reshape);
	
	initBitmap(DIM, DIM);
	
	glutMainLoop();
}

