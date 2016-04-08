Quickjoin
=========

an algorithm to solve the similarity join problem (serial version), implemented according to the paper:

Edwin H. Jacox, Hanan Samet:
Metric space similarity joins. ACM Trans. Database Syst. 33(2) (2008)

* C++ implementation.
* Object oriented programming.
* Pivots are selected randomly.

OpenMPQuickjoin
=========

a parallel algorithm to solve the similarity join problem (OpenMP version), implemented according to the paper:

Shichao Jin, Okhee Kim, Wenya Feng:
Accelerating Metric Space Similarity Joins with Multi-core and Many-core Processors. ICCSA (5) 2013: 166-180

* C++ implementation.
* Object oriented programming.
* Pivots are selected randomly.
* OpenMP is used exploiting the new feature of OpenMP 3.0, that is, "task".

CUDAQuickjoin
=============
a parallel algorithm to solve the similarity join problem exploiting the massive power of a GPU (CUDA version), implemented according to the paper:

Shichao Jin, Okhee Kim, Wenya Feng:
Accelerating Metric Space Similarity Joins with Multi-core and Many-core Processors. ICCSA (5) 2013: 166-180

* C++ implementation.
* Object oriented programming.
* Pivots are selected randomly.
* tested on CUDA 5.0, compute capability 2.1.