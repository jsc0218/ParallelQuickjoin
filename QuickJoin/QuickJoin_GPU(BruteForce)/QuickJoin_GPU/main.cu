#include <iostream>
#include <fstream>
#include <string>
#include "QuickJoin.h"
using namespace std;

const int blockSize = 32;
const int gridSizeX = 10000;

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

__host__ int main(int argc, char** argv) 
{
	Object *h_objs = new Object[DBSize];
	ifstream fin(DBPath.c_str());
	for (int i=0; i<DBSize; ++i) 
	{
		Object *obj = Read(fin);
		h_objs[i] = *obj;
		delete obj;
	}
	fin.close();

	findCudaDevice(argc, (const char **)argv);
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	
	Object *d_objs;
	checkCudaErrors(cudaMalloc(&d_objs, sizeof(Object)*DBSize));
	for (int i=0; i<DBSize; ++i)
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
	delete[] h_objs;
	
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	long long size = DBSize;
	long long mulSize = size * size;
	int blockNum = (int)ceil(mulSize / (float)blockSize);
	int gridSizeY = (int)ceil(blockNum / (float)gridSizeX);
	dim3 gridSize(gridSizeX, gridSizeY);
	NestedLoop<<<gridSize, blockSize, 0>>>(d_objs, 0, DBSize-1);
	checkCudaErrors(cudaDeviceSynchronize());

	sdkStopTimer(&timer);
	cout<<"Processing time: "<<sdkGetTimerValue(&timer)<<" (ms)"<<endl;
	sdkDeleteTimer(&timer);
	
	checkCudaErrors(cudaFree(d_objs));
	cudaDeviceReset();
	exit(0);
}

