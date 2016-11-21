#include <cstdio>
#include <algorithm>

#include <string.h>

#include "sort.h"

#include <pthread.h>

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray
{
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size)
	{
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin()
	{
		return buffer;
	}

	iterator end()
	{
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static
void
cxx_sort(int *array, size_t size)
{
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static
void
simple_quicksort(int *array, size_t size)
{
	int pivot, pivot_count, i;
	int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1)
	{
		// Bad, bad way to pick a pivot
		// Better take a sample and pick
		// it median value.
		pivot = array[size / 2];
		
		left = (int*)malloc(size * sizeof(int));
		right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = 0; i < size; i++)
		{
			if(array[i] < pivot)
			{
				left[left_size] = array[i];
				left_size++;
			}
			else if(array[i] > pivot)
			{
				right[right_size] = array[i];
				right_size++;
			}
			else
			{
				pivot_count++;
			}
		}

		// Recurse		
		simple_quicksort(left, left_size);
		simple_quicksort(right, right_size);

		// Merge
		memcpy(array, left, left_size * sizeof(int));
		for(i = left_size; i < left_size + pivot_count; i++)
		{
			array[i] = pivot;
		}
		memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		// Free
		free(left);
		free(right);
	}
	else
	{
		// Do nothing
	}
}

struct thread_args
{
	int* subarray;
	size_t size;
};

void
Parallel_merge_sort(int* array, size_t size)
{
malloc(sizeof(thread_args))
	if(size == 1)
		return NULL;

	else if(size == 2)
	{
		// Compare elements
		if(data->subarray[size] > data->subarray[size+1])
		{	
			int temp = data->subarray;
			data->subarray[size] = data->subarray[size+1];
			data->subarray[size+1] = temp;
		}
		return NULL;
	}

	else
	{
		int* left, right;
		int left_size, right_size;
		int split_size = size / 2;
		int left_ind = 0, right_ind = 0;
		int count = 0;


		left = data->subarray;
		left_size = split_size;

		right = data->subarray; + split_size;
		right_size = split_size + size % 2;

		Parallel_merge_sort(left, left_size)
		Parallel_merge_sort(right, right_size);

		do {
			if(left[left_ind] <= right[right_ind])
			{
				//buffers[buf_usage][count] = left[left_ind];
				left_ind++;
			}
			else
			{
				//buffers[buf_usage][count] = right[right_ind];
				right_ind++;
			}
			//buf_usage = !buf_usage;
			++count;

		} while(right_size > right_ind && left_size > left_ind);

		int i;
		for (i = left_ind; i < left_size; ++i)
		{
			//buffers[buf_usage][count] = left[i];
			++count;
		}
		for (i = right_ind; i < right_size; ++i)
		{
			//buffers[buf_usage][count] = right[i];
			++count;
		}
/*
		if(!buf_usage)
		{
			int* temp_ptr;
			temp_ptr = data->subarray;
			data->subarray = data->buffer;
			//data->buffer = temp_ptr;
		}*/
	}
}

void
merge_sort(int* array, size_t size)
{
#if NB_THREADS == 0

Parallel_merge_sort(array, size);

#else

int step_size = size / NB_THREADS;
pthread thread[NB_THREADS];
thread_args t_args;

t_args.size = step_size;


// invalidate cache line?
int i;
for (i = 0; i < NB_THREADS - 1; ++i)
{
	t_args.subarray = array + step_size*i;
	pthread_create(&thread, NULL, Parallel_merge_sort, &t_args);
}

// Speciall case for the last thread to handle the last if size in unevenly devided by NB_THREADS
t_args.subarray = array + step_size*i+1;
t_args.size = step_size + size % NB_THREADS;
pthread_create(&thread, NULL, Parallel_merge_sort, &t_args);

for (int i = 0; i < NB_THREADS; ++i)
{
	pthread_join(thread[i], NULL);
}

return NULL;

#endif
}


// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void
sort(int* array, size_t size)
{
	// Do some sorting magic here. Just remember: if NB_THREADS == 0, then everything must be sequential
	// When this function returns, all data in array must be sorted from index 0 to size and not element
	// should be lost or duplicated.

	// Use preprocessor directives to influence the behavior of your implementation. For example NB_THREADS denotes
	// the number of threads to use and is defined at compareile time. NB_THREADS == 0 denotes a sequential version.
	// NB_THREADS == 1 is a parallel version using only one thread that can be useful to monitor the overhead
	// brought by addictional parallelization code.



	// This is to make the base skeleton to work. Replace it with your own implementation
	simple_quicksort(array, size);




	// Alternatively, use C++ sequential sort, just to see how fast it is
	//cxx_sort(array, size);

	// Note: you are NOT allowed to demonstrate code that uses C or C++ standard sequential or parallel sort or merge
	// routines (qsort, std::sort, std::merge, etc). It's more interesting to learn by writing it yourself.



	// Reproduce this structure here and there in your code to compile sequential or parallel versions of your code.
#if NB_THREADS == 0
	// Some sequential-specific sorting code
#else
	// Some parallel sorting-related code
#endif // #if NB_THREADS
}

