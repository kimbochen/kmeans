#ifndef KMEANS_H
#define KMEANS_H

#include <cublas_v2.h>
#include <cuda_fp16.h>

#define ADDR(i, j, dim0) ( (j) * (dim0) + (i) )  // Column major
#define CEIL_DIV(a, b) (a + b - 1) / b

#ifndef N_ITERS
#define N_ITERS 1000
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 32
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (BLOCK_DIM * BLOCK_DIM)
#endif


/* CUDA */
void cudaKMeans(
    float *means, float *points, int *clr_sizes, int *clr_idxs,
    int n_clrs, int n_pts, int n_dims,
    float *out_means
);

__global__ void findNearestCluster(
    float *means, float *points, int *clr_idxs, int n_clrs, int n_pts, int n_dims
);

__global__ void computeNewMeans(
    float *means, float *points, int *clr_idxs, int *clr_sizes,
    int n_dims, int n_clrs, int n_pts
);

__device__ float squaredDist(float *q, float *r, int q_idx, int r_idx, int n_dim);



/* cuBLAS */
void cublasKMeans(
    float *means, float *points, int *clr_sizes,
    int *clr_idxs, int n_clrs, int n_pts, int n_dims,
    float *out_means
);

void pairwiseSquaredDist(
    cublasHandle_t handle, dim3 *grid_p, dim3 *block_p,
    float *q, float *r, float *sqnorm_q, float *sqnorm_r, float *pw_sqdist,
    int n_q, int n_r, int n_d
);

__global__ void addSquaredNorms(
    float *pw_sqdist, float *sqnorm_q, float *sqnorm_r, int n_q, int n_r
);

__global__ void argMin(float *arr, int n_rows, int n_cols, int *idxs);

__global__ void makeAvgMatrix(float *avg, int *clr_idxs, int *clr_sizes, int n_pts, int n_clrs);


/* Tensor Core */
void tcuKMeans(
    float *means, float *points, int *clr_sizes, int *clr_idxs,
    int n_clrs, int n_pts, int n_dims, float *out_means
);

__global__ void float2Half(float *arr_sg, __half *arr_hf, int n_elems);

__global__ void half2Float(__half *arr_hf, float *arr_sg, int n_elems);

void pairwiseSquaredDistHalf(
    cublasHandle_t handle, dim3 *grid_p, dim3 *block_p,
    __half *q, __half *r, __half *sqnorm_q, __half *sqnorm_r, __half *pw_sqdist,
    int n_q, int n_r, int n_d
);

__global__ void addSquaredNormsHalf(
    __half *pw_sqdist, __half *sqnorm_q, __half *sqnorm_r, int n_q, int n_r
);

__global__ void argMinHalf(__half *arr, int n_rows, int n_cols, int *idxs);

__global__ void makeAvgMatrixHalf(__half *avg, int *clr_idxs, int *clr_sizes, int n_pts, int n_clrs);


/* Utilities */
void computeInertia(float *means, float *points, int *clr_idxs, int n_clrs, int n_pts, int n_dims);

void print2DArray(float *arr, int dim0, int dim1);

#endif
