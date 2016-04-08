#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "Object.h"

__constant__ double eps = 0;
__device__ int resultCount = 0;

__global__ void NestedLoop(Object *objs, int start, int end);
