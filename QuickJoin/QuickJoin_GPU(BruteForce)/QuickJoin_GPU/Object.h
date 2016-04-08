#pragma once

#ifdef VECTOR

#include <cmath>

const int dimension = 112;

class Object
{
public:
	__host__ __device__ Object()
	{
		x = NULL;
	}
	__host__ __device__ Object(const double *buf)
	{
		x = new double[dimension];
		for (int i=0; i<dimension; ++i) 
		{
			x[i] = buf[i];
		}
	}
	__host__ __device__ Object(const Object& obj)
	{
		x = new double[dimension];
		for (int i=0; i<dimension; ++i) 
		{
			x[i] = obj.x[i];
		}
	}
	__host__ __device__ ~Object()
	{
		if (x)
		{
			delete[] x;
		}
	}

	__host__ __device__ void operator=(const Object& obj)
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
	__host__ __device__ double Distance(const Object& obj) const
	{
		double dist = 0;
		for (int i=0; i<dimension; ++i) 
		{
			dist += pow(x[i]-obj.x[i], 2);
		}
		return sqrt(dist);
	}

//private:
	double *x;
};

#endif


#ifdef STRING

const int stringMaxLen = 21; 

class Object
{
public:
	__host__ __device__ Object() 
	{ 
		x = NULL; 
		length = 0; 
	}
	__host__ __device__ Object(const char *buf, int len)
	{
		length = len;
		x = new char[length];
		for (int i=0; i<length; ++i) 
		{
			x[i] = buf[i];
		}
	}
	__host__ __device__ Object(const Object& obj)
	{
		length = obj.length;
		x = new char[length];
		for (int i=0; i<length; ++i) 
		{
			x[i] = obj.x[i];
		}
	}
	__host__ __device__ ~Object()
	{
		if (x)
		{
			delete[] x;
		}
	}
	
	__host__ __device__ void operator=(const Object& obj)
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
	__host__ __device__ int Distance(const Object& obj) const
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

//private:
	char *x;
	int length;
};

#endif

