#pragma once

#include <istream>
using namespace std;

#ifdef VECTOR

extern int dimension;

class Object
{
public:
	Object();  
	Object(const double *buf);
	Object(const Object& obj);
	~Object();

	void operator=(const Object& obj);
	double Distance(const Object& obj) const;

private:
	double *x;
};

Object *Read(istream& in);  // read an object from standard input or a file
#endif

#ifdef STRING

class Object
{
public:
	Object();  
	Object(const char *buf, int len);
	Object(const Object& obj);
	~Object();
	
	void operator=(const Object& obj);
	int Distance(const Object& obj) const;  

private:
	char *x;
	int length;
};

Object *Read(istream& in);  // read an object from standard input or a file
#endif
