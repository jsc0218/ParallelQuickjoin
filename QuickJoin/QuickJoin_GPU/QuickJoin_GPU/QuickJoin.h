#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "Object.h"
#include "Stack.h"

__constant__ double eps = 1.0;
const int constSmallNumber = 50000;
__device__ int resultCount = 0;
struct __align__(16) Offset
{
	Offset(int _posL=0, int _posR=0, int _count=0, int _posWinL=0, int _posWinR=0)
	{
		posL = _posL;
		posR = _posR;
		count = _count;
		posWinL = _posWinL;
		posWinR = _posWinR;
	}
	~Offset()
	{
	}
	int posL;
	int posR;
	int count;
	int posWinL;
	int posWinR;
};

#ifdef VECTOR
struct SharedObject
{
	__device__ double Distance(const Object& obj) const
	{
		double dist = 0;
		for (int i=0; i<dimension; ++i) 
		{
			dist += pow(x[i]-obj.x[i], 2);
		}
		return sqrt(dist);
	}

	double x[dimension];
};
#endif

#ifdef STRING
struct SharedObject
{
	__device__ int Distance(const Object& obj) const
	{
		int rec0[stringMaxLen+1];
		int rec1[stringMaxLen+1];
		int length2 = obj.length;
		for (int j=0; j<=length2; ++j) 
		{
			rec0[j] = j;
		}

		for (int i=0; i<length; ++i) 
		{
			if ((i+1) & 1)
			{
				rec1[0] = i+1;
				for (int j=0; j<length2; ++j) 
				{
					rec1[j+1] = ((x[i] == obj.x[j]) ? rec0[j]: min(min(rec0[j+1], rec1[j]), rec0[j])+1);
				}
			} 
			else
			{
				rec0[0] = i+1;
				for (int j=0; j<length2; ++j) 
				{
					rec0[j+1] = ((x[i] == obj.x[j]) ? rec1[j]: min(min(rec1[j+1], rec0[j]), rec1[j])+1);
				}
			}
		}

		return (length&1) ? rec1[obj.length]: rec0[obj.length];
	}

	char x[stringMaxLen];
	int length;
};
#endif

__global__ void NestedLoop(Object *objs, int *idx, int start, int end);
__global__ void QuickJoin(Object *objs, Stack *stack, int *in, int *out, Offset *offset, int start, int end, int r1, int r2, int *winL, int *winR);
__global__ void NestedLoopWin(Object *objs, int *idx1, int start1, int end1, int *idx2, int start2, int end2);
__global__ void QuickJoinWin(Object *objs, Stack *stack, int *in1, int *in2, int *out1, int *out2, Offset *offset1, Offset *offset2, int start1, int end1, int start2, int end2, int r1, int r2, int *winL1, int *winR1, int *winL2, int *winR2);
__global__ void Synchronize();
