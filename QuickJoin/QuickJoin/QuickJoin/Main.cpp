#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "Object.h"
using namespace std;

#ifdef STRING
string path = "C:\\English.dic";
int amount = 69069;
int constSmallNumber1 = 70;
int constSmallNumber2 = 70;
#endif

#ifdef VECTOR
string path = "C:\\colors_112_112682.ascii";
int amount = 112682;
int constSmallNumber1 = 120;
int constSmallNumber2 = 120;
int dimension = 112;
#endif

double eps = 0.0707;
Object *objs = new Object[amount];
int *idx = new int[amount];
int resultCount = 0;
void NestedLoop(int start, int end)
{
	for (int i=start; i<=end; ++i)
	{
		for (int j=i+1; j<=end; ++j)
		{
			if (objs[idx[i]].Distance(objs[idx[j]]) <= eps)
			{
				++resultCount;
				//cout<<idx[i]<<" <--> "<<idx[j]<<endl;
			}
		}
	}
}

void NestedLoop2(int *idxA, int startA, int endA, int *idxB, int startB, int endB)
{
	for (int i=startA; i<=endA; ++i)
	{
		for (int j=startB; j<=endB; ++j)
		{
			if (objs[idxA[i]].Distance(objs[idxB[j]]) <= eps)
			{
				++resultCount;
				//cout<<idxA[i]<<" <--> "<<idxB[j]<<endl;
			}
		}
	}
}

void Partition(int *idx, int startIdx, int endIdx, int pIdx1, int pIdx2, int& partIdx, int *&winL, int& sizeL, int *&winG, int& sizeG)
{
	double r = objs[pIdx1].Distance(objs[pIdx2]);
	double startDist = objs[idx[startIdx]].Distance(objs[pIdx1]);
	double endDist = objs[idx[endIdx]].Distance(objs[pIdx1]);
	vector<int> winLIdx, winGIdx;
	while (startIdx <= endIdx)
	{
		while (endDist > r)
		{
			if (endDist <= r+eps)
			{
				winGIdx.push_back(idx[endIdx]);
			}
			if ((--endIdx) >= startIdx)
			{
				endDist = objs[idx[endIdx]].Distance(objs[pIdx1]);
			}
			else
			{
				break;
			}
		}
		while (startDist <= r)
		{
			if (startDist >= r-eps)
			{
				winLIdx.push_back(idx[startIdx]);
			}
			if ((++startIdx) <= endIdx)
			{
				startDist = objs[idx[startIdx]].Distance(objs[pIdx1]);
			} 
			else
			{
				break;
			}
		}

		if (startIdx < endIdx)
		{
			if (endDist >= r-eps)
			{
				winLIdx.push_back(idx[endIdx]);
			}
			if (startDist <= r+eps)
			{
				winGIdx.push_back(idx[startIdx]);
			}
			swap(idx[startIdx], idx[endIdx]);
			startDist = objs[idx[++startIdx]].Distance(objs[pIdx1]);
			endDist = objs[idx[--endIdx]].Distance(objs[pIdx1]);
		}
	}
	partIdx = endIdx;

	sizeL = (int)winLIdx.size();
	sizeG = (int)winGIdx.size();
	winL = (sizeL == 0) ? NULL : new int[sizeL];
	winG = (sizeG == 0) ? NULL : new int[sizeG];
	for (int i=0; i<sizeL; ++i)
	{
		winL[i] = winLIdx[i];
	}
	for (int i=0; i<sizeG; ++i)
	{
		winG[i] = winGIdx[i];
	}
}

void QuickJoinWin(int *idx1, int start1, int end1, int *idx2, int start2, int end2)
{
	if (end1<start1 || end2<start2)
	{
		return;
	}

	int size1 = end1 - start1 + 1;
	int size2 = end2 - start2 + 1;
	int totalLen = size1 + size2;
	if (totalLen < constSmallNumber2)
	{
		NestedLoop2(idx1, start1, end1, idx2, start2, end2);
		return;
	}

	int randNum = rand()%totalLen;
	int p1 = (randNum<size1) ? idx1[randNum+start1] : idx2[randNum-size1+start2];
	int p2;
	do 
	{
		randNum = rand()%totalLen;
		p2 = (randNum<size1) ? idx1[randNum+start1] : idx2[randNum-size1+start2];
	} while (p1 == p2);

	int partIdx1, partIdx2, sizeL1, sizeG1, sizeL2, sizeG2;
	int *winL1, *winG1, *winL2, *winG2;
	Partition(idx1, start1, end1, p1, p2, partIdx1, winL1, sizeL1, winG1, sizeG1);
	Partition(idx2, start2, end2, p1, p2, partIdx2, winL2, sizeL2, winG2, sizeG2);

	QuickJoinWin(winL1, 0, sizeL1-1, winG2, 0, sizeG2-1);
	QuickJoinWin(winG1, 0, sizeG1-1, winL2, 0, sizeL2-1);
	QuickJoinWin(idx1, start1, partIdx1, idx2, start2, partIdx2);
	QuickJoinWin(idx1, partIdx1+1, end1, idx2, partIdx2+1, end2);

	if (winL1) 
	{
		delete[] winL1;
	}
	if (winG1) 
	{
		delete[] winG1;
	}
	if (winL2) 
	{
		delete[] winL2;
	}
	if (winG2)
	{
		delete[] winG2;
	}
}

void QuickJoin(int start, int end)
{
	if (end < start)
	{
		return;
	}

	int size = end - start + 1;
	if (size < constSmallNumber1)
	{
		NestedLoop(start, end);
		return;
	}

	int p1 = idx[rand()%size + start];
	int p2 = idx[rand()%size + start];
	while (p1 == p2)
	{
		p2 = idx[rand()%size + start];
	}

	int *winL, *winG, partIdx, sizeL, sizeG;
	Partition(idx, start, end, p1, p2, partIdx, winL, sizeL, winG, sizeG);

	QuickJoinWin(winL, 0, sizeL - 1, winG, 0, sizeG - 1);
	QuickJoin(start, partIdx);
	QuickJoin(partIdx+1, end);

	if (winL) 
	{
		delete[] winL;
	}
	if (winG)
	{
		delete[] winG;
	}
}

int main()
{
	ifstream fin(path.c_str());
	for (int i=0; i<amount; ++i) 
	{
		Object *obj = Read(fin);
		objs[i] = *obj;
		idx[i] = i;
		delete obj;
	}
	fin.close();

	clock_t clock_start, clock_end;  
	clock_start = clock();  
	QuickJoin(0, amount-1);
	clock_end = clock();  
	cout<<(clock_end-clock_start)<<endl;  
	
	if (objs)
	{
		delete[] objs;
	}
	if (idx) 
	{
		delete[] idx;
	}
	cout<<resultCount<<endl;
	return 0;
}