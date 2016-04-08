#include "Object.h"
#include <string>
using namespace std;

#ifdef VECTOR
#include <cmath>

Object::Object()
{
	x = NULL;
}

Object::Object(const double *buf) 
{ 
	x = new double[dimension];
	for (int i=0; i<dimension; ++i) 
	{
		x[i] = buf[i];
	}
}

Object::Object(const Object& obj)
{
	x = new double[dimension];
	for (int i=0; i<dimension; ++i) 
	{
		x[i] = obj.x[i];
	}
}

Object::~Object()
{
	if (x)
	{
		delete[] x;
	}
}

void Object::operator=(const Object& obj)
{
	if (!x)
	{
		x = new double[dimension];
	}
	for (int i=0; i<dimension; ++i)
	{
		x[i] = obj.x[i];
	}
}

double Object::Distance(const Object& obj) const
{
	double dist = 0;
	for (int i=0; i<dimension; ++i) 
	{
		dist += pow(x[i]-obj.x[i], 2);
	}
	return sqrt(dist);
}

Object *Read(istream& in)
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

int EditDistance(const char* strA, int lenA, const char* strB, int lenB)
{
	int **rec = new int *[2];
	rec[0] = new int[lenB+1];
	rec[1] = new int[lenB+1];
	for (int j=0; j<lenB+1; ++j) 
	{
		rec[0][j] = (int)j;
	}

	for (int i=0; i<lenA; ++i) 
	{
		rec[(i+1)%2][0] = (int)(i+1);
		for (int j=0; j<lenB; ++j) 
		{
			rec[(i+1)%2][j+1] = ((strA[i] == strB[j]) ? rec[i%2][j]: min(min(rec[i%2][j+1], rec[(i+1)%2][j]), rec[i%2][j])+1);
		}
	}

	int result = rec[lenA%2][lenB];
	delete[] rec[0];
	delete[] rec[1];
	delete[] rec;
	return result;
}

Object::Object()
{
	x = NULL;
	length = 0;
}

Object::Object(const char *buf, int len) 
{
	length = len;
	x = new char[length];
	for (int i=0; i<length; ++i) 
	{
		x[i] = buf[i];
	}
}

Object::Object(const Object& obj)
{
	length = obj.length;
	x = new char[length];
	for (int i=0; i<length; ++i) 
	{
		x[i] = obj.x[i];
	}
}

Object::~Object()
{
	if (x)
	{
		delete[] x;
	}
}

void Object::operator=(const Object& obj)
{
	if (length != obj.length)
	{
		if (x)
		{
			delete[] x;
		}
		length = obj.length;
		x = new char[length];
	}
	for (int i=0; i<length; ++i)
	{
		x[i] = obj.x[i];
	}
}

int Object::Distance(const Object& obj) const
{
	return EditDistance(x, length, obj.x, obj.length);
}

Object *Read(istream& in)
{
	string cmdLine;
	getline(in, cmdLine);
	Object *obj = new Object(cmdLine.c_str(), (int)cmdLine.size());
	return obj;
}
#endif