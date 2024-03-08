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
#define TILE_WIDTH 16  // Define the width of the tile


//-----------------------------------------------------------------------------
__global__ void kernelFunc(const float *f, const float *g, float *result, int n) {
    
    __shared__ float tile_f[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_g[TILE_WIDTH][TILE_WIDTH];
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float output = 0.0f;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    // Loop over all tiles
    for (int t = 0; t < numTiles; ++t) {
        // Load one tile of f and g into shared memory
        int tiledRow = t * TILE_WIDTH + ty;
        int tiledCol = t * TILE_WIDTH + tx;

        if (tiledRow < n && tiledCol < n) {
            tile_f[ty][tx] = f[tiledRow * n + tiledCol];
            tile_g[ty][tx] = g[tiledRow * n + tiledCol];
        } else {
            tile_f[ty][tx] = 0.0f;
            tile_g[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < n && col < n) {
            for (int i = 0; i < TILE_WIDTH; ++i) {
                output += tile_f[ty][i] * tile_g[i][tx];
            }
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        result[row * n + col] = output;
    }
}