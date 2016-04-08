#include <iostream>
#include <fstream>
#include <ctime>
#include "Object.h"
#include <omp.h>
using namespace std;

#ifdef STRING
string path = "C:\\English.dic";
int amount = 69069;
#endif

#ifdef VECTOR
string path = "C:\\colors_112_112682.ascii";
int amount = 112682;
int dimension = 112;
#endif

double eps = 0;
Object *objs = new Object[amount];
int resultCount = 0;
void NestedLoop(int start, int end)
{
	#pragma omp parallel for schedule(dynamic)
	for (int i=start; i<=end; ++i)
	{
		for (int j=i+1; j<=end; ++j)
		{
			if (objs[i].Distance(objs[j]) <= eps)
			{
				++resultCount;
				//cout<<i<<" <--> "<<j<<endl;
			}
		}
	}
}

int main()
{
	ifstream fin(path.c_str());
	for (int i=0; i<amount; ++i) 
	{
		Object *obj = Read(fin);
		objs[i] = *obj;
		delete obj;
	}
	fin.close();

	clock_t clock_start, clock_end;  
	clock_start = clock();  
	NestedLoop(0, amount-1);
	clock_end = clock();  
	cout<<(clock_end-clock_start)<<endl;  

	if (objs)
	{
		delete[] objs;
	}
	cout<<resultCount<<endl;
	return 0;
}