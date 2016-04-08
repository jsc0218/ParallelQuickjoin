#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <pthread.h>
#include "QuickJoin.h"
using namespace std;

extern const int DBSize;
extern const int threadCount;
const int activeThread = threadCount + 1;
const int winCount = 1024; 

typedef pair<int, int> SizePair;

bool SizePairCompareFunc(SizePair x, SizePair y) 
{
	return (x.second < y.second); 
}

class MemAllocator
{
public:
	MemAllocator()
	{
		bitmap = new int[activeThread];
		pthread_mutex_init(&lock, NULL);

		in1s = new int*[activeThread];
		in2s = new int*[activeThread];
		out1s = new int*[activeThread];
		out2s = new int*[activeThread];

		d_stacks = new Stack*[activeThread];
		h_stacks = new Stack*[activeThread];

		offsets = new Offset*[activeThread];

		winLs1 = new int**[activeThread];
		winRs1 = new int**[activeThread];
		winLs2 = new int**[activeThread];
		winRs2 = new int**[activeThread];
		for (int i=0; i<activeThread; ++i)
		{
			bitmap[i] = 0;

			checkCudaErrors(cudaMalloc(&in1s[i], sizeof(int)*DBSize));
			checkCudaErrors(cudaMalloc(&in2s[i], sizeof(int)*DBSize));
			checkCudaErrors(cudaMalloc(&out1s[i], sizeof(int)*DBSize));
			checkCudaErrors(cudaMalloc(&out2s[i], sizeof(int)*DBSize));
			
			checkCudaErrors(cudaMalloc(&d_stacks[i], sizeof(Stack)));
			checkCudaErrors(cudaMallocHost(&h_stacks[i], sizeof(Stack)));

			checkCudaErrors(cudaMalloc(&offsets[i], sizeof(Offset)*winCount*2));
			
			winLs1[i] = new int*[winCount];
			winRs1[i] = new int*[winCount];
			winLs2[i] = new int*[winCount];
			winRs2[i] = new int*[winCount];
			for (int j=0; j<winCount; ++j)
			{
				int count = (int) (ceil((float)DBSize / (j+1)) * sizeof(int));
				checkCudaErrors(cudaMalloc(&winLs1[i][j], count));
				checkCudaErrors(cudaMalloc(&winRs1[i][j], count));
				checkCudaErrors(cudaMalloc(&winLs2[i][j], count));
				checkCudaErrors(cudaMalloc(&winRs2[i][j], count));
			}
		}
	}

	~MemAllocator()
	{
		delete[] bitmap;
		pthread_mutex_destroy(&lock);

		for (int i=0; i<activeThread; ++i)
		{
			checkCudaErrors(cudaFree(in1s[i]));
			checkCudaErrors(cudaFree(in2s[i]));
			checkCudaErrors(cudaFree(out1s[i]));
			checkCudaErrors(cudaFree(out2s[i]));

			checkCudaErrors(cudaFree(d_stacks[i]));
			checkCudaErrors(cudaFreeHost(h_stacks[i]));

			checkCudaErrors(cudaFree(offsets[i]));

			for (int j=0; j<winCount; ++j)
			{
				checkCudaErrors(cudaFree(winLs1[i][j]));
				checkCudaErrors(cudaFree(winRs1[i][j]));
				checkCudaErrors(cudaFree(winLs2[i][j]));
				checkCudaErrors(cudaFree(winRs2[i][j]));
			}
			delete[] winLs1[i];
			delete[] winRs1[i];
			delete[] winLs2[i];
			delete[] winRs2[i];
		}
		delete[] in1s;
		delete[] in2s;
		delete[] out1s;
		delete[] out2s;

		delete[] d_stacks;
		delete[] h_stacks;

		delete[] offsets;

		delete[] winLs1;
		delete[] winRs1;
		delete[] winLs2;
		delete[] winRs2;
	}

	void AllocateInOut(int bitIndex, int** _in1, int** _out1, int** _in2=NULL, int** _out2=NULL)
	{
		*_in1 = in1s[bitIndex];
		*_out1 = out1s[bitIndex];
		if (_in2 && _out2)
		{
			*_in2 = in2s[bitIndex];
			*_out2 = out2s[bitIndex];
		}
	}

	void AllocateStack(int bitIndex, Stack** _d_stack, Stack **_h_stack)
	{
		*_d_stack = d_stacks[bitIndex];
		*_h_stack = h_stacks[bitIndex];
	}

	void AllocateOffset(int bitIndex, Offset** _offsets)
	{
		*_offsets = offsets[bitIndex];
	}

	void AllocateWin(int bitIndex, vector<SizePair>& sizePairVec, int** _winLs1, int** _winRs1, int** _winLs2 = NULL, int** _winRs2 = NULL)
	{
		sort(sizePairVec.begin(), sizePairVec.end(), SizePairCompareFunc);
		for (unsigned int i=0; i<sizePairVec.size(); ++i)
		{
			_winLs1[sizePairVec[sizePairVec.size()-1-i].first] = winLs1[bitIndex][i];
			_winRs1[sizePairVec[sizePairVec.size()-1-i].first] = winRs1[bitIndex][i];
			if (_winLs2 && _winRs2)
			{
				_winLs2[sizePairVec[sizePairVec.size()-1-i].first] = winLs2[bitIndex][i];
				_winRs2[sizePairVec[sizePairVec.size()-1-i].first] = winRs2[bitIndex][i];
			}
		}
	}

	int AllocateBitmap()
	{
		int index = -1;
		pthread_mutex_lock(&lock);
		for (int i=0; i<activeThread; ++i)
		{
			if (bitmap[i] == 0)
			{
				bitmap[i] = 1;
				index = i;
				break;
			}
		}
		pthread_mutex_unlock(&lock);
		return index;
	}

	void DeallocateBitmap(int index)
	{
		bitmap[index] = 0;
	}

private:
	int	*bitmap;
	pthread_mutex_t lock;
	int **in1s;
	int **in2s;
	int **out1s;
	int **out2s;
	Stack **d_stacks;
	Stack **h_stacks;
	Offset **offsets;
	int ***winLs1;
	int	***winRs1;
	int ***winLs2;
	int	***winRs2;
};