#include "QuickJoin.h"

__global__ void NestedLoop(Object *objs, int *idx, int start, int end)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = end - start + 1;
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

	if (objs[idx[start+i]].Distance(objs[idx[start+j]]) <= eps)
	{
		++resultCount;
		//printf("%d <--> %d\n", idx[start+i], idx[start+j]);
	}
}

__global__ void QuickJoin(Object *objs, Stack *stack, int *in, int *out, Offset *offset, int start, int end, int r1, int r2, int *winL, int *winR)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = end - start + 1;
	if (thread_id >= size)
	{
		return;
	}
	
	int count = 0;
	int data = in[start+thread_id];

	__shared__ double r;
	__shared__ SharedObject pivotObj;
	__shared__ int countL;
	__shared__ int countR;
	__shared__ int posL;
	__shared__ int posR;
	__shared__ int countWinL;
	__shared__ int countWinR;
	__shared__ int posWinL;
	__shared__ int posWinR;
	if (threadIdx.x == 0)
	{
		countL = countR = posL = posR = countWinL = countWinR = posWinL = posWinR = 0;
		int p1 = in[start+r1];
		int p2 = in[start+r2];
		#ifdef VECTOR
		for (int i=0; i<dimension; ++i)
		{
			pivotObj.x[i] = objs[p1].x[i];
		}
		#endif
		#ifdef STRING
		pivotObj.length = objs[p1].length;
		for (int i=0; i<pivotObj.length; ++i)
		{
			pivotObj.x[i] = objs[p1].x[i];
		}
		#endif
		r = pivotObj.Distance(objs[p2]);
	}
	__syncthreads();

	double dist = pivotObj.Distance(objs[data]);
	if (dist <= r)
	{
		atomicAdd(&countL, 1);
		if (dist >= r-eps)
		{
			atomicAdd(&countWinL, 1);
		}
	} 
	else
	{
		atomicAdd(&countR, 1);
		if (dist <= r+eps)
		{
			atomicAdd(&countWinR, 1);
		}
	}
	__syncthreads();

	__shared__ int offsetL;
	__shared__ int offsetR;
	__shared__ int offsetWinL;
	__shared__ int offsetWinR;
	if (threadIdx.x == 0)
	{
		offsetL = start + atomicAdd(&(offset->posL), countL);
		offsetR = end - (atomicAdd(&(offset->posR), countR) + countR) + 1;
		count = countL + countR;
		offsetWinL = atomicAdd(&(offset->posWinL), countWinL);
		offsetWinR = atomicAdd(&(offset->posWinR), countWinR);
	}
	__syncthreads();

	if (dist <= r)
	{
		out[offsetL+atomicAdd(&posL,1)] = data;
		if (dist >= r-eps)
		{
			winL[offsetWinL+atomicAdd(&posWinL,1)] = data;
		}
	} 
	else
	{
		out[offsetR+atomicAdd(&posR,1)] = data;
		if (dist <= r+eps)
		{
			winR[offsetWinR+atomicAdd(&posWinR,1)] = data;
		}
	}

	if (threadIdx.x==0 && atomicAdd(&(offset->count),count)+count==size)
	{
		int part = end - offset->posR;
		if (start < part)
		{
			if (!stack->Push(StackElement(start, part)))
			{
				printf("Stack overfull!\n");
			}
		}
		if (part+1 < end)
		{
			if (!stack->Push(StackElement(part+1, end)))
			{
				printf("Stack overfull!\n");
			}
		}
		if (offset->posWinL>0 && offset->posWinR>0)
		{
			if (!stack->Push(StackElement((int)winL, (int)winR, 0, offset->posWinL-1, 0, offset->posWinR-1)))
			{
				printf("Stack overfull!\n");
			}
		}
	}
}

__global__ void NestedLoopWin(Object *objs, int *idx1, int start1, int end1, int *idx2, int start2, int end2)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = end1 - start1 + 1;
	int size2 = end2 - start2 + 1;
	if (thread_id >= size1*size2)
	{
		return;
	}

	int i = thread_id / size2;
	int j = thread_id - i*size2;

	if (objs[idx1[start1+i]].Distance(objs[idx2[start2+j]]) <= eps)
	{
		++resultCount;
		//printf("%d <--> %d\n", idx1[start1+i], idx2[start2+j]);
	}
}

__global__ void QuickJoinWin(Object *objs, Stack *stack, int *in1, int *in2, int *out1, int *out2, Offset *offset1, Offset *offset2, int start1, int end1, int start2, int end2, int r1, int r2, int *winL1, int *winR1, int *winL2, int *winR2)
{
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int size1 = end1 - start1 + 1;
	int size2 = end2 - start2 + 1;
	int size = size1 + size2;
	if (thread_id >= size)
	{
		return;
	}

	int count = 0;
	int data = (thread_id<size1) ? in1[start1+thread_id] : in2[start2+thread_id-size1];
	
	__shared__ double r;
	__shared__ SharedObject pivotObj;
	__shared__ int countL1;
	__shared__ int countR1;
	__shared__ int countL2;
	__shared__ int countR2;
	__shared__ int posL1;
	__shared__ int posR1;
	__shared__ int posL2;
	__shared__ int posR2;
	__shared__ int countWinL1;
	__shared__ int countWinR1;
	__shared__ int countWinL2;
	__shared__ int countWinR2;
	__shared__ int posWinL1;
	__shared__ int posWinR1;
	__shared__ int posWinL2;
	__shared__ int posWinR2;
	if (threadIdx.x == 0)
	{
		countL1 = countR1 = countL2 = countR2 = posL1 = posR1 = posL2 = posR2 = countWinL1 = countWinR1 = countWinL2 = countWinR2 = posWinL1 = posWinR1 = posWinL2 = posWinR2 = 0;
		int p1 = (r1<size1) ? in1[start1+r1] : in2[start2+r1-size1];
		int p2 = (r2<size1) ? in1[start1+r2] : in2[start2+r2-size1];	
		#ifdef VECTOR
		for (int i=0; i<dimension; ++i)
		{
			pivotObj.x[i] = objs[p1].x[i];
		}
		#endif
		#ifdef STRING
		pivotObj.length = objs[p1].length;
		for (int i=0; i<pivotObj.length; ++i)
		{
			pivotObj.x[i] = objs[p1].x[i];
		}
		#endif
		r = pivotObj.Distance(objs[p2]);
	}
	__syncthreads();

	double dist = pivotObj.Distance(objs[data]);
	if (dist <= r)
	{
		(thread_id<size1) ? atomicAdd(&countL1, 1) : atomicAdd(&countL2, 1);
		if (dist >= r-eps)
		{
			(thread_id<size1) ? atomicAdd(&countWinL1, 1) : atomicAdd(&countWinL2, 1);
		}
	} 
	else
	{
		(thread_id<size1) ? atomicAdd(&countR1, 1) : atomicAdd(&countR2, 1);
		if (dist <= r+eps)
		{
			(thread_id<size1) ? atomicAdd(&countWinR1, 1) : atomicAdd(&countWinR2, 1);
		}
	}
	__syncthreads();

	__shared__ int offsetL1;
	__shared__ int offsetR1;
	__shared__ int offsetL2;
	__shared__ int offsetR2;
	__shared__ int offsetWinL1;
	__shared__ int offsetWinR1;
	__shared__ int offsetWinL2;
	__shared__ int offsetWinR2;
	if (threadIdx.x == 0)
	{
		offsetL1 = start1 + atomicAdd(&(offset1->posL), countL1);
		offsetR1 = end1 - (atomicAdd(&(offset1->posR), countR1) + countR1) + 1;
		offsetL2 = start2 + atomicAdd(&(offset2->posL), countL2);
		offsetR2 = end2 - (atomicAdd(&(offset2->posR), countR2) + countR2) + 1;
		count = countL1 + countR1 + countL2 + countR2;
		offsetWinL1 = atomicAdd(&(offset1->posWinL), countWinL1);
		offsetWinR1 = atomicAdd(&(offset1->posWinR), countWinR1);
		offsetWinL2 = atomicAdd(&(offset2->posWinL), countWinL2);
		offsetWinR2 = atomicAdd(&(offset2->posWinR), countWinR2);
	}
	__syncthreads();

	if (dist <= r)
	{
		if (thread_id < size1)
		{
			out1[offsetL1+atomicAdd(&posL1,1)] = data;
		}
		else
		{
			out2[offsetL2+atomicAdd(&posL2,1)] = data;
		}
		if (dist >= r-eps)
		{
			if (thread_id < size1)
			{
				winL1[offsetWinL1+atomicAdd(&posWinL1,1)] = data;
			}
			else
			{
				winL2[offsetWinL2+atomicAdd(&posWinL2,1)] = data;
			}
		}
	} 
	else
	{
		if (thread_id < size1)
		{
			out1[offsetR1+atomicAdd(&posR1,1)] = data;
		} 
		else
		{
			out2[offsetR2+atomicAdd(&posR2,1)] = data;
		}
		if (dist <= r+eps)
		{
			if (thread_id < size1)
			{
				winR1[offsetWinR1+atomicAdd(&posWinR1,1)] = data;
			}
			else
			{
				winR2[offsetWinR2+atomicAdd(&posWinR2,1)] = data;
			}
		}
	}

	if (threadIdx.x==0 && atomicAdd(&(offset1->count),count)+count==size)
	{
		int part1 = end1 - offset1->posR;
		int part2 = end2 - offset2->posR;
		if (start1<=part1 && start2<=part2)
		{
			if (!stack->Push(StackElement(0, 0, start1, part1, start2, part2)))
			{
				printf("Stack overfull!\n");
			}
		}
		if (part1+1<=end1 && part2+1<=end2)
		{
			if (!stack->Push(StackElement(0, 0, part1+1, end1, part2+1, end2)))
			{
				printf("Stack overfull!\n");
			}
		}
		if (offset1->posWinL>0 && offset2->posWinR>0)
		{
			if (!stack->Push(StackElement((int)winL1, (int)winR2, 0, offset1->posWinL-1, 0, offset2->posWinR-1)))
			{
				printf("Stack overfull!\n");
			}
		}
		if (offset1->posWinR>0 && offset2->posWinL>0)
		{
			if (!stack->Push(StackElement((int)winR1, (int)winL2, 0, offset1->posWinR-1, 0, offset2->posWinL-1)))
			{
				printf("Stack overfull!\n");
			}
		}
	}
}

__global__ void Synchronize()
{

}