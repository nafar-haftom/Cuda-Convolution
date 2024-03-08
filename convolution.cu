#include "convolution.h"

#define ty threadIdx.y
#define tz threadIdx.z
#define tx threadIdx.x

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
// you may define other functions here!


//-----------------------------------------------------------------------------
__global__ void kernelFunc(const float *f, const float *g, float *result, int n) {
    int resultSize = n + n - 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    register float sum;
    if (row < resultSize && col < resultSize) {
        sum = 0.0f;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int fRow = row - i;
                int fCol = col - j;

                // Check boundaries of the input signal f
                if (fRow >= 0 && fRow < n && fCol >= 0 && fCol < n) {
                    sum += f[fRow * n + fCol] * g[i * n + j];
                }
            }
        }

        result[row * resultSize + col] = sum;
    }
}
