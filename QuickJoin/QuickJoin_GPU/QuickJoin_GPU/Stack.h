#pragma once

const int MaxStackSize = 4096;

class StackElement
{
public:
	__host__ __device__ StackElement(int _posL=-1, int _posR=-1, int _startL=-1, int _endL=-1, int _startR=-1, int _endR=-1)
	{
		posL = _posL;
		posR = _posR;
		startL = _startL;
		endL = _endL;
		startR = _startR;
		endR = _endR;
	}
	__host__ __device__ StackElement(const StackElement& elem)
	{
		posL = elem.posL; 
		posR = elem.posR;
		startL = elem.startL;
		endL = elem.endL;
		startR = elem.startR;
		endR = elem.endR;
	}
	__host__ __device__ ~StackElement()
	{
	}
	__host__ __device__ void operator=(const StackElement& elem)
	{
		posL = elem.posL; 
		posR = elem.posR;
		startL = elem.startL;
		endL = elem.endL;
		startR = elem.startR;
		endR = elem.endR;
	}

//private:
	int posL;
	int posR;
	int startL;
	int endL;
	int startR;
	int endR;
};

class Stack
{
public:
	__host__ __device__ Stack()
	{
		top = -1;
	}
	__host__ __device__ ~Stack()
	{
	}
	__device__ bool Push(const StackElement& elem)
	{
		int idx = atomicAdd(&top, 1);
		if (idx < MaxStackSize-1) 
		{
			elements[++idx] = elem;
			return true;
		}
		return false;
	}
	__host__ bool Pop(StackElement& elem)
	{
		if (top == -1)
		{
			return false;
		}
		elem = elements[top--];
		return true;
	}
	__host__ bool Top(StackElement& elem) const
	{
		if (top == -1)
		{
			return false;
		}
		elem = elements[top];
		return true;
	}

private:
	int top;
	StackElement elements[MaxStackSize];
};
