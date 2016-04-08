#include "QuickJoin.h"

__global__ void NestedLoop(Object *objs, int start, int end)
{
	long long thread_id = (long long)blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	long long size = end - start + 1;
	if (thread_id >= size*size)
	{
		return;
	}

	int i = thread_id / size;
	int j = thread_id - i*size;
	if (i <= j)
	{
		return;
	}

	if (objs[start+i].Distance(objs[start+j]) <= eps)
	{
		++resultCount;
		//printf("%d <--> %d\n", start+i, start+j);
	}
}