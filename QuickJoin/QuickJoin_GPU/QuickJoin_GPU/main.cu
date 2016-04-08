#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "MemAllocator.h"
#include "threadpool.h"
using namespace std;

#pragma comment(lib, "pthreadVC2.lib")  

threadpool_t *threadPool;
const int threadCount = 1;
const int queueSize = 51200000;

const int blockSize = 64;

MemAllocator *memAllocator;

int *gTop;

struct KernelParameter
{
	KernelParameter(StackElement _elem, int _r1, int _r2)
	{
		elem = _elem;
		r1 = _r1;
		r2 = _r2;
	}
	~KernelParameter()
	{
	}
	StackElement elem;
	int r1;
	int r2;
};

struct ThreadParameter
{
	ThreadParameter(StackElement _elem, Object *_d_obj=NULL, int *_winL=NULL, int *_winR=NULL)
	{
		elem = _elem;
		d_obj = _d_obj;
		winL = _winL;
		winR = _winR;
	}
	~ThreadParameter()
	{
	}
	StackElement elem;
	Object *d_obj;
	int *winL;
	int *winR;
};

#ifdef VECTOR
const string DBPath = "I:\\colors_112_112682.ascii";
const int DBSize = 112682;
__host__ Object *Read(istream& in)
{
	string cmdLine;
	double *x = new double[dimension];
	for (int i=0; i<dimension; ++i) 
	{
		in>>cmdLine;
		x[i] = atof(cmdLine.c_str());
	}
	Object *obj = new Object(x);
	delete[] x;
	return obj;
}
#endif

#ifdef STRING
const string DBPath = "I:\\English.dic";
const int DBSize = 69069;
__host__ Object *Read(istream& in)
{
	string cmdLine;
	getline(in, cmdLine);
	Object *obj = new Object(cmdLine.c_str(), (int)cmdLine.size());
	return obj;
}
#endif

__host__ void QuickJoinWinLaunch(void *parameter)
{
	int bitIndex = memAllocator->AllocateBitmap();
	if (bitIndex == -1)
	{
		cout<<"Resource allocation failed!"<<endl;
	}

	vector<cudaStream_t> streams(1);
	checkCudaErrors(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));

	Stack *d_stack, *h_stack;
	memAllocator->AllocateStack(bitIndex, &d_stack, &h_stack);
	checkCudaErrors(cudaMemcpyAsync(d_stack, gTop, sizeof(int), cudaMemcpyHostToDevice, streams[0]));

	Offset *d_offsets;
	memAllocator->AllocateOffset(bitIndex, &d_offsets);
	checkCudaErrors(cudaMemsetAsync(d_offsets, 0, sizeof(Offset)*2, streams[0]));

	int size1 = (((ThreadParameter *)parameter)->elem).endL + 1;
	int size2 = (((ThreadParameter *)parameter)->elem).endR + 1;
	int size = size1 + size2;
	int	r1 = rand() % size;
	int	r2 = rand() % size;
	while (r1 == r2)
	{
		r2 = rand() % size;
	}
	
	int *outL, *outR, *posL, *posR;
	memAllocator->AllocateInOut(bitIndex, &posL, &outL, &posR, &outR);
	checkCudaErrors(cudaMemcpyAsync(posL, ((ThreadParameter *)parameter)->winL, sizeof(int)*size1, cudaMemcpyHostToDevice, streams[0]));
	checkCudaErrors(cudaMemcpyAsync(posR, ((ThreadParameter *)parameter)->winR, sizeof(int)*size2, cudaMemcpyHostToDevice, streams[0]));
	delete[] ((ThreadParameter *)parameter)->winL;
	delete[] ((ThreadParameter *)parameter)->winR;

	vector<KernelParameter *> paras;
	paras.push_back(new KernelParameter(StackElement(((ThreadParameter *)parameter)->elem), r1, r2));
	Object *d_objs = ((ThreadParameter *)parameter)->d_obj;
	delete parameter;

	vector<SizePair> sizePairVec;
	sizePairVec.push_back(SizePair(0, max(size1, size2)));
	int **winLs1 = new int*[1];
	int **winRs1 = new int*[1];
	int **winLs2 = new int*[1];
	int **winRs2 = new int*[1];
	memAllocator->AllocateWin(bitIndex, sizePairVec, winLs1, winRs1, winLs2, winRs2);

	int	gridSize = (int)ceil(size / ((float)blockSize));
	QuickJoinWin<<<gridSize, blockSize, 0, streams[0]>>>(d_objs, d_stack, posL, posR, outL, outR, d_offsets, d_offsets+1, 0, (paras[0]->elem).endL, 0, (paras[0]->elem).endR, r1, r2, winLs1[0], winRs1[0], winLs2[0], winRs2[0]);
	sizePairVec.resize(0);
	delete[] winLs1;
	delete[] winRs1;
	delete[] winLs2;
	delete[] winRs2;

	int turn = 0;
	int *in1 = posL;
	int *in2 = posR;
	int *out1 = outL;
	int *out2 = outR;
	while (1)
	{
		int stackSize = (min(streams.size()*4, MaxStackSize) * 6 + 1) * sizeof(int);
		for (unsigned int i=0; i<streams.size(); ++i)
		{
			checkCudaErrors(cudaStreamSynchronize(streams[i]));
			checkCudaErrors(cudaStreamDestroy(streams[i]));
			delete paras[i];
		}
		streams.resize(0);
		paras.resize(0);

		cudaStream_t s;
		streams.push_back(s);
		checkCudaErrors(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
		checkCudaErrors(cudaMemcpyAsync(h_stack, d_stack, stackSize, cudaMemcpyDeviceToHost, streams[0]));
		checkCudaErrors(cudaMemcpyAsync(d_stack, gTop, sizeof(int), cudaMemcpyHostToDevice, streams[0]));
		Synchronize<<<0, 0, 0, streams[0]>>>();
		checkCudaErrors(cudaStreamSynchronize(streams[0]));
		checkCudaErrors(cudaStreamDestroy(streams[0]));
		streams.resize(0);

		StackElement elem;
		if (!h_stack->Top(elem))
		{
			break;
		}
		vector<cudaStream_t> winStreams;
		while (h_stack->Pop(elem))
		{
			if (elem.posL==0 && elem.posR==0)
			{
				cudaStream_t s;
				streams.push_back(s);
				checkCudaErrors(cudaStreamCreateWithFlags(&streams[streams.size()-1], cudaStreamNonBlocking));

				long long size1 = elem.endL - elem.startL + 1;
				long long size2 = elem.endR - elem.startR + 1;
				if (size1*size2 <= constSmallNumber)
				{
					r1 = 0;
					r2 = 0;
				}
				else
				{
					int size = size1 + size2;
					r1 = rand() % size;
					r2 = rand() % size;
					while (r1 == r2)
					{
						r2 = rand() % size;
					}
				}
				paras.push_back(new KernelParameter(elem, r1, r2));

				sizePairVec.push_back(SizePair(streams.size()-1, max(size1, size2)));
			}
			else
			{
				long long size1 = elem.endL + 1;
				long long size2 = elem.endR + 1;
				long long mulSize = size1 * size2;
				if (mulSize <= constSmallNumber)
				{
					cudaStream_t s;
					winStreams.push_back(s);
					checkCudaErrors(cudaStreamCreateWithFlags(&winStreams[winStreams.size()-1], cudaStreamNonBlocking));
					gridSize = (int)ceil(mulSize / (float)blockSize);
					NestedLoopWin<<<gridSize, blockSize, 0, winStreams[winStreams.size()-1]>>>(d_objs, (int *)elem.posL, 0, elem.endL, (int *)elem.posR, 0, elem.endR);
				}
				else
				{
					int *winL = new int[size1];
					int *winR = new int[size2];
					cudaStream_t s;
					checkCudaErrors(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
					checkCudaErrors(cudaMemcpyAsync(winL, (int *)elem.posL, sizeof(int)*size1, cudaMemcpyDeviceToHost, s));
					checkCudaErrors(cudaMemcpyAsync(winR, (int *)elem.posR, sizeof(int)*size2, cudaMemcpyDeviceToHost, s));
					Synchronize<<<0, 0, 0, s>>>();
					checkCudaErrors(cudaStreamSynchronize(s));
					checkCudaErrors(cudaStreamDestroy(s));
					if (threadpool_add(threadPool, &QuickJoinWinLaunch, new ThreadParameter(elem, d_objs, winL, winR)))
					{
						delete[] winL;
						delete[] winR;
						cout<<"Failed in thread creation!"<<endl;
						return;
					}
				}
			}
		}

		if (streams.size() > 0)
		{
			checkCudaErrors(cudaMemsetAsync(d_offsets, 0, sizeof(Offset)*streams.size()*2, streams[0]));
		}

		for (unsigned int i=0; i<winStreams.size(); ++i)
		{
			checkCudaErrors(cudaStreamSynchronize(winStreams[i]));
			checkCudaErrors(cudaStreamDestroy(winStreams[i]));
		}

		winLs1 = new int*[streams.size()];
		winRs1 = new int*[streams.size()];
		winLs2 = new int*[streams.size()];
		winRs2 = new int*[streams.size()];
		memAllocator->AllocateWin(bitIndex, sizePairVec, winLs1, winRs1, winLs2, winRs2);

		++turn;
		in1 = turn%2 ? outL : posL;
		in2 = turn%2 ? outR : posR;
		out1 = turn%2 ? posL : outL;
		out2 = turn%2 ? posR : outR;
		for (unsigned int i=0; i<streams.size(); ++i)
		{
			long long size1 = (paras[i]->elem).endL - (paras[i]->elem).startL + 1;
			long long size2 = (paras[i]->elem).endR - (paras[i]->elem).startR + 1;
			long long mulSize = size1 * size2;
			if (mulSize <= constSmallNumber)
			{
				gridSize = (int)ceil(mulSize / (float)blockSize);
				NestedLoopWin<<<gridSize, blockSize, 0, streams[i]>>>(d_objs, in1, (paras[i]->elem).startL, (paras[i]->elem).endL, in2, (paras[i]->elem).startR, (paras[i]->elem).endR);
			}
			else
			{
				gridSize = (int)ceil((size1 + size2) / (float)blockSize);
				QuickJoinWin<<<gridSize, blockSize, 0, streams[i]>>>(d_objs, d_stack, in1, in2, out1, out2, d_offsets+i*2, d_offsets+1+i*2, (paras[i]->elem).startL, (paras[i]->elem).endL, (paras[i]->elem).startR, (paras[i]->elem).endR, paras[i]->r1, paras[i]->r2, winLs1[i], winRs1[i], winLs2[i], winRs2[i]);
			}
		}
		sizePairVec.resize(0);
		delete[] winLs1;
		delete[] winRs1;
		delete[] winLs2;
		delete[] winRs2;
	}

	memAllocator->DeallocateBitmap(bitIndex);
}

__host__ void QuickJoinLaunch(Object *h_objs, Object *d_objs, int amount)
{
	threadPool = threadpool_create(threadCount, queueSize);
	
	memAllocator = new MemAllocator;
	int bitIndex = memAllocator->AllocateBitmap();
	if (bitIndex == -1)
	{
		cout<<"Resource allocation failed!"<<endl;
	}

	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	vector<cudaStream_t> streams(1);
	checkCudaErrors(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));

	int *i_idx, *o_idx;
	memAllocator->AllocateInOut(bitIndex, &i_idx, &o_idx);
	int *h_idx = new int[amount];
	for (int i=0; i<amount; ++i)
	{
		h_idx[i] = i;
	}
	checkCudaErrors(cudaMemcpyAsync(i_idx, h_idx, sizeof(int)*amount, cudaMemcpyHostToDevice, streams[0]));
	delete[] h_idx;
	
	Stack *d_stack, *h_stack;
	memAllocator->AllocateStack(bitIndex, &d_stack, &h_stack);
	checkCudaErrors(cudaMemcpyAsync(d_stack, gTop, sizeof(int), cudaMemcpyHostToDevice, streams[0]));

	Offset *d_offsets;
	memAllocator->AllocateOffset(bitIndex, &d_offsets);
	checkCudaErrors(cudaMemsetAsync(d_offsets, 0, sizeof(Offset), streams[0]));
	
	int r1 = rand() % amount;
	int r2 = rand() % amount;
	while (r1 == r2)
	{
		r2 = rand() % amount;
	}
	vector<KernelParameter *> paras;
	paras.push_back(new KernelParameter(StackElement(0, amount-1), r1, r2));

	vector<SizePair> sizePairVec;
	sizePairVec.push_back(SizePair(0, amount));
	int **winLs = new int*[1];
	int **winRs = new int*[1];
	memAllocator->AllocateWin(bitIndex, sizePairVec, winLs, winRs);

	int gridSize = (int)ceil(((float)amount)/blockSize);
	QuickJoin<<<gridSize, blockSize, 0, streams[0]>>>(d_objs, d_stack, i_idx, o_idx, d_offsets, 0, amount-1, r1, r2, winLs[0], winRs[0]);
	sizePairVec.resize(0);
	delete[] winLs;
	delete[] winRs;

	int turn = 0;
	int *in = i_idx;
	int *out = o_idx;
	while (1)
	{
		int stackSize = (min(streams.size()*3, MaxStackSize) * 6 + 1) * sizeof(int);
		for (unsigned int i=0; i<streams.size(); ++i)
		{
			checkCudaErrors(cudaStreamSynchronize(streams[i]));
			checkCudaErrors(cudaStreamDestroy(streams[i]));
			delete paras[i];
		}
		streams.resize(0);
		paras.resize(0);

		cudaStream_t s;
		streams.push_back(s);
		checkCudaErrors(cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking));
		checkCudaErrors(cudaMemcpyAsync(h_stack, d_stack, stackSize, cudaMemcpyDeviceToHost, streams[0]));
		checkCudaErrors(cudaMemcpyAsync(d_stack, gTop, sizeof(int), cudaMemcpyHostToDevice, streams[0]));
		Synchronize<<<0, 0, 0, streams[0]>>>();
		checkCudaErrors(cudaStreamSynchronize(streams[0]));
		checkCudaErrors(cudaStreamDestroy(streams[0]));
		streams.resize(0);

		StackElement elem;
		if (!h_stack->Top(elem))
		{
			break;
		}
		vector<cudaStream_t> winStreams;
		while (h_stack->Pop(elem))
		{
			if (elem.startL==0 && elem.endL!=-1 && elem.startR==0 && elem.endR!=-1)
			{
				long long size1 = elem.endL + 1;
				long long size2 = elem.endR + 1;
				long long mulSize = size1 * size2;
				if (mulSize <= constSmallNumber)
				{
					cudaStream_t s;
					winStreams.push_back(s);
					checkCudaErrors(cudaStreamCreateWithFlags(&winStreams[winStreams.size()-1], cudaStreamNonBlocking));
					gridSize = (int)ceil(mulSize / ((float)blockSize));
					NestedLoopWin<<<gridSize, blockSize, 0, winStreams[winStreams.size()-1]>>>(d_objs, (int *)elem.posL, 0, elem.endL, (int *)elem.posR, 0, elem.endR);
				}
				else
				{
					int *winL = new int[size1];
					int *winR = new int[size2];
					cudaStream_t s;
					checkCudaErrors(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
					checkCudaErrors(cudaMemcpyAsync(winL, (int *)elem.posL, sizeof(int)*size1, cudaMemcpyDeviceToHost, s));
					checkCudaErrors(cudaMemcpyAsync(winR, (int *)elem.posR, sizeof(int)*size2, cudaMemcpyDeviceToHost, s));
					Synchronize<<<0, 0, 0, s>>>();
					checkCudaErrors(cudaStreamSynchronize(s));
					checkCudaErrors(cudaStreamDestroy(s));
					if (threadpool_add(threadPool, &QuickJoinWinLaunch, new ThreadParameter(elem, d_objs, winL, winR)))
					{
						delete[] winL;
						delete[] winR;
						cout<<"Failed in thread creation!"<<endl;
						return;
					}
				}
			} 
			else
			{
				cudaStream_t s;
				streams.push_back(s);
				checkCudaErrors(cudaStreamCreateWithFlags(&streams[streams.size()-1], cudaStreamNonBlocking));

				int size = elem.posR - elem.posL + 1;
				r1 = rand() % size;
				r2 = rand() % size;
				while (r1 == r2)
				{
					r2 = rand() % size;
				}
				paras.push_back(new KernelParameter(elem, r1, r2));

				sizePairVec.push_back(SizePair(streams.size()-1, size));
			}
		}

		if (streams.size() > 0)
		{
			checkCudaErrors(cudaMemsetAsync(d_offsets, 0, sizeof(Offset)*streams.size(), streams[0]));
		}

		for (unsigned int i=0; i<winStreams.size(); ++i)
		{
			checkCudaErrors(cudaStreamSynchronize(winStreams[i]));
			checkCudaErrors(cudaStreamDestroy(winStreams[i]));
		}

		winLs = new int*[streams.size()];
		winRs = new int*[streams.size()];
		memAllocator->AllocateWin(bitIndex, sizePairVec, winLs, winRs);

		++turn;
		in = turn%2 ? o_idx : i_idx;
		out = turn%2 ? i_idx : o_idx;
		for (unsigned int i=0; i<streams.size(); ++i)
		{
			long long size = (paras[i]->elem).posR - (paras[i]->elem).posL + 1;
			long long mulSize = size * size;
			if ((mulSize-size)>>1 <= constSmallNumber)
			{
				gridSize = (int)ceil(mulSize / (float)blockSize);
				NestedLoop<<<gridSize, blockSize, 0, streams[i]>>>(d_objs, in, (paras[i]->elem).posL, (paras[i]->elem).posR);
			}
			else
			{
				gridSize = (int)ceil(size / (float)blockSize);
				QuickJoin<<<gridSize, blockSize, 0, streams[i]>>>(d_objs, d_stack, in, out, d_offsets+i, (paras[i]->elem).posL, (paras[i]->elem).posR, paras[i]->r1, paras[i]->r2, winLs[i], winRs[i]);	
			}
		}
		sizePairVec.resize(0);
		delete[] winLs;
		delete[] winRs;
	}

	memAllocator->DeallocateBitmap(bitIndex);

	while (!threadpool_destroy_ready(threadPool))
	{
	}

	sdkStopTimer(&timer);
	cout<<"Processing time: "<<sdkGetTimerValue(&timer)<<" (ms)"<<endl;
	sdkDeleteTimer(&timer);

	delete memAllocator;
	threadpool_destroy(threadPool);
}

__host__ int main(int argc, char** argv) 
{
	int amount = DBSize;
	Object *h_objs = new Object[amount];
	ifstream fin(DBPath.c_str());
	for (int i=0; i<amount; ++i) 
	{
		Object *obj = Read(fin);
		h_objs[i] = *obj;
		delete obj;
	}
	fin.close();

	findCudaDevice(argc, (const char **)argv);
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	checkCudaErrors(cudaMallocHost(&gTop, sizeof(int)));
	*gTop = -1;
	
	Object *d_objs;
	checkCudaErrors(cudaMalloc(&d_objs, sizeof(Object)*amount));
	for (int i=0; i<amount; ++i)
	{
		#ifdef VECTOR
		double *tmp;
		checkCudaErrors(cudaMalloc(&tmp, sizeof(double)*dimension));
		checkCudaErrors(cudaMemcpy(tmp, h_objs[i].x, sizeof(double)*dimension, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&(d_objs[i].x), &tmp, sizeof(double *), cudaMemcpyHostToDevice));
		#endif
		
		#ifdef STRING
		char *tmp;
		checkCudaErrors(cudaMalloc(&tmp, sizeof(char)*h_objs[i].length));
		checkCudaErrors(cudaMemcpy(tmp, h_objs[i].x, sizeof(char)*h_objs[i].length, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&(d_objs[i].x), &tmp, sizeof(char *), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(&(d_objs[i].length), &(h_objs[i].length), sizeof(int), cudaMemcpyHostToDevice));
		#endif
	}
	
	QuickJoinLaunch(h_objs, d_objs, amount);

	delete[] h_objs;
	checkCudaErrors(cudaFree(d_objs));
	checkCudaErrors(cudaFreeHost(gTop));
	cudaDeviceReset();
	exit(0);
}

