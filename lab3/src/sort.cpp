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

void copy(int* a1, int* a2, size_t size)
{
	int i;
	for(i = 0; i < size; ++i)
	{
	  a1[i] = a2[i];
	}
}

void merge(int* array, size_t size1, size_t size2)
{
	
	int* temp = (int*)malloc(sizeof(int) * (size1 + size2));
	int ind1 = 0;
	int ind2 = 0;

	// merge the sub arrays
	while (size1 > ind1 && size2 > ind2) {
		if(array[ind1] < array[size1 + ind2])
		{
			temp[ind1 + ind2] = array[ind1];
			++ind1;
		}
		else
		{
			temp[ind1 + ind2] = array[size1 + ind2];
			++ind2;
		}
	}

	// insert the remaining elements
	for(; ind1 < size1; ++ind1)
		temp[ind1 + ind2] = array[ind1];

	for(; ind2 < size2; ++ind2)
		temp[ind1 + ind2] = array[size1 + ind2];

	copy(array, temp, size1 + size2);
	//memcpy(array, &temp, sizeof(int) * size1 + size2);

	free(temp);
}

void merge_sort(int* array, size_t size)
{	
	if(size == 1)
	{
		return;
	}
	else if(size == 2)
	{
		// Compare elements
		if(array[0] > array[1])
		{	
			int temp = array[0];
			array[0] = array[1];
			array[1] = temp;
		}
		return;
	}
	else
	{
		int* left; 
		int* right;
		int left_size, right_size;
		int split_size = size / 2;
		int left_ind = 0, right_ind = 0;
		int count = 0;

		left = array;
		left_size = split_size;

		right = array + split_size;
		right_size = size - split_size;

		merge_sort(left, left_size);
		merge_sort(right, right_size);

		merge(array, left_size, right_size);
	}
}

void* parallel_merge_sort(void* args)
{
	struct thread_args* args_t = (thread_args*)args;
	merge_sort(args_t->subarray, args_t->size);
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
	//simple_quicksort(array, size);


#if NB_THREADS == 0
	merge_sort(array, size);
#else
	struct thread_args t_args;
	
	int step_size = size / NB_THREADS;
	pthread_t thread[NB_THREADS];
	
	t_args.size = step_size;

	// invalidate cache line?
	int i;
	for (i = 0; i < NB_THREADS - 1; ++i)
	{
		t_args.subarray = array + step_size*i;
		pthread_create(&thread[i], NULL, parallel_merge_sort, (void*)&t_args);
	}

	// Speciall case for the last thread to handle the last if size in unevenly devided by NB_THREADS
	t_args.subarray = array + step_size*i;
	t_args.size = size - step_size * (NB_THREADS - 1);
	pthread_create(&thread[i], NULL, parallel_merge_sort, (void*)&t_args);

	for (int i = 0; i < NB_THREADS; ++i)
	{
		pthread_join(thread[i], NULL);
	}

	// merge the threads
#if NB_THREADS == 2
	merge(array, step_size, size - step_size);
#elif NB_THREADS == 3
	merge(array, step_size, step_size);
	merge(array, step_size * 2, size - step_size*2);
#elif NB_THREADS == 4
	merge(array, step_size, step_size);
	merge(array + step_size*2, step_size, size - step_size*3);
	merge(array, step_size * 2, size - step_size * 2);
#endif
	return;

#endif

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

