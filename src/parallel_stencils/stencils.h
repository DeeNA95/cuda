#include "../cx.h"

template <typename T>
__global__ void stencil2d(cr_Ptr<T> a, r_Ptr<T> b, int nx, int ny);
template <typename T>
T array_diff_max(cr_Ptr<T> a, cr_Ptr<T> b, int nx, int ny);
