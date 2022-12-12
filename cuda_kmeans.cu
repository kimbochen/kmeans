#include <stdio.h>  // puts
#include <time.h>  // clock

#include <cuda.h>
#include "kmeans.h"


void cudaKMeans(
    float *means, float *points, int *clr_sizes, int *clr_idxs,
    int n_clrs, int n_pts, int n_dims,
    float *out_means
)
{
#ifndef BMK
    puts("----- Benchmarking CUDA version -----");
#endif

    clock_t start = clock();

    // Constants
    int means_dsize = n_dims * n_clrs * sizeof(float);
    int points_dsize = n_dims * n_pts * sizeof(float);
    int clr_idxs_dsize = n_pts * sizeof(int);
    int clr_sizes_dsize = n_clrs * sizeof(int);

    // Allocate memory
    float *means_dev, *points_dev;
    int *clr_idxs_dev, *clr_sizes_dev;

    cudaMalloc((void**)&means_dev, means_dsize);
    cudaMalloc((void**)&points_dev, points_dsize);
    cudaMalloc((void**)&clr_idxs_dev, clr_idxs_dsize);
    cudaMalloc((void**)&clr_sizes_dev, clr_sizes_dsize);

    // Load data into device
    cudaMemcpy(points_dev, points, points_dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(means_dev, out_means, means_dsize, cudaMemcpyHostToDevice);

    // Kernel config
    const int grid_size = CEIL_DIV(n_pts, THREADS_PER_BLOCK);
    // dim3 grid_m(CEIL_DIV(n_dims, BLOCK_DIM), CEIL_DIV(n_clrs, BLOCK_DIM));
    // dim3 block(BLOCK_DIM, BLOCK_DIM);

    for (int it = 0; it < N_ITERS; it++) {
        cudaMemcpy(means_dev, out_means, means_dsize, cudaMemcpyHostToDevice);
        memset(clr_sizes, 0, clr_sizes_dsize);
        for (int i = 0; i < n_dims * n_clrs; i++) {
            out_means[i] = 0.0;
        }

        findNearestCluster<<<grid_size, THREADS_PER_BLOCK>>>(
            means_dev, points_dev, clr_idxs_dev, n_clrs, n_pts, n_dims
        );
        cudaDeviceSynchronize();
        cudaMemcpy(clr_idxs, clr_idxs_dev, clr_idxs_dsize, cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_pts; i++) {
            clr_sizes[clr_idxs[i]]++;
        }

        // For every element (i, j) in a means vector:
        //   For every index clr_idx == clr_idxs[k], if clr_idx == j, add points(k, j) to means(i, j)
        //   Divide means(i, j) by clr_sizes[j]

        // cudaMemcpy(clr_sizes_dev, clr_sizes, clr_sizes_dsize, cudaMemcpyHostToDevice);
        // computeNewMeans<<<grid_m, block>>>(
        //     means_dev, points_dev, clr_idxs_dev, clr_sizes_dev, n_dims, n_clrs, n_pts
        // );
        // cudaDeviceSynchronize();

        // for (int j = 0; j < n_clrs; j++) {
        //     for (int i = 0; i < n_dims; i++) {
        //         for (int k = 0; k < n_pts; k++) {
        //             if (clr_idxs[k] == j) {
        //                 out_means[ADDR(i, j, n_dims)] += points[ADDR(i, k, n_dims)];
        //             }
        //         }
        //         out_means[ADDR(i, j, n_dims)] /= clr_sizes[j];
        //     }
        // }

        for (int j = 0; j < n_pts; j++) {
            int clr_idx = clr_idxs[j];
            for (int i = 0; i < n_dims; i++) {
                out_means[ADDR(i, clr_idx, n_dims)] += points[ADDR(i, j, n_dims)];
            }
        }
        for (int j = 0; j < n_clrs; j++) {
            for (int i = 0; i < n_dims; i++) {
                out_means[ADDR(i, j, n_dims)] /= clr_sizes[j];
            }
        }
    }
    // cudaMemcpy(out_means, means_dev, means_dsize, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

#ifdef BMK
    printf("%.4f\n", elapsed);
#else
    printf("Time elapsed: %.3f seconds.\n", elapsed);
#endif

    cudaFree(means_dev);
    cudaFree(points_dev);
    cudaFree(clr_idxs_dev);
    cudaFree(clr_sizes_dev);
}


__global__ void findNearestCluster(
    float *means, float *points, int *clr_idxs, int n_clrs, int n_pts, int n_dims
)
{
    int pt_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (pt_idx < n_pts) {
        float min_dist = squaredDist(means, points, 0, pt_idx, n_dims);
        int arg_min = 0;

        for (int i = 1; i < n_clrs; i++) {
            float dist = squaredDist(means, points, i, pt_idx, n_dims);
            if (dist < min_dist) {
                min_dist = dist;
                arg_min = i;
            }
        }

        clr_idxs[pt_idx] = arg_min;
    }
}

__global__ void computeNewMeans(
    float *means, float *points, int *clr_idxs, int *clr_sizes,
    int n_dims, int n_clrs, int n_pts
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < n_dims && j < n_pts) {
        means[ADDR(i, j, n_dims)] = 0.0;

        for (int k = 0; k < n_pts; k++) {
            if (clr_idxs[k] == j) {
                means[ADDR(i, j, n_dims)] += points[ADDR(i, k, n_dims)];
            }
        }

        means[ADDR(i, j, n_dims)] /= clr_sizes[j];
    }
}


__device__ float squaredDist(float *q, float *r, int q_idx, int r_idx, int n_dims)
{
    float sqdist = 0.0;

    for (int i = 0; i < n_dims; i++) {
        float diff = q[ADDR(i, q_idx, n_dims)] - r[ADDR(i, r_idx, n_dims)];
        sqdist += diff * diff;
    }

    return sqdist;
}
