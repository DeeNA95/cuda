#pragma once

// Undefine min/max to avoid conflicts
#undef min
#undef max

// CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "helper_cuda.h"

// Thrust includes
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
//#include "thrust/system/cuda/experimental/pinned_allocator.h"
//#include "thrust.h"

// C++ standard includes
#include <algorithm>
#include <float.h>  // for _controlfp_s
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Math defines
#define _USE_MATH_DEFINES

// Constants
#define ESC 27  // for OpenCV GUI etc (ASCII escape)

// Type aliases
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using ullong = unsigned long long;
using llong = long long;

// Const versions
using cuchar = const unsigned char;
using cushort = const unsigned short;
using cuint = const unsigned int;
using culong = const unsigned long;
using cullong = const unsigned long long;
using cchar = const char;
using cshort = const short;
using cint = const int;
using cfloat = const float;
using clong = const long;
using cdouble = const double;
using cllong = const long long;

// CUDA-specific const types
using cint3 = const int3;
using cfloat3 = const float3;

// // Pointer templates
template<typename T> using r_Ptr = T* __restrict__;  // pointer variable, data variable
template<typename T> using cr_Ptr = const T* __restrict__;  // pointer variable, data constant
template<typename T> using cvr_Ptr = T* const __restrict__;  // pointer constant, data variable
template<typename T> using ccr_Ptr = const T* const __restrict__;  // pointer constant, data constant

// // Thrust vector aliases
//template<typename T> using thrustHvecPin = thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>;
template<typename T> using thrustHvec = thrust::host_vector<T>;
template<typename T> using thrustDvec = thrust::device_vector<T>;

// // Utility function to get raw pointer from thrust device vector
// template<typename T> T* trDptr(thrustDvec<T>& a) {
//     return a.data().get();
// }

