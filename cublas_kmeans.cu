#include <stdio.h>  // puts
#include <time.h>  // clock

#include <cuda.h>
#include <cublas_v2.h>
#include "kmeans.h"


void cublasKMeans(
    float *means, float *points, int *clr_sizes, int *clr_idxs,
    int n_clrs, int n_pts, int n_dims, float *out_means
)
{
#ifndef BMK
    puts("----- Benchmarking cuBLAS version -----");
#endif

    clock_t start = clock();

    // Arrays
    float *means_dev, *points_dev;
    float *means_sqnorm, *points_sqnorm, *pw_sqdist;
    int *clr_idxs_dev, *clr_sizes_dev;
    float *avg;

    // Constants
    int means_dsize = n_dims * n_clrs * sizeof(float);
    int points_dsize = n_dims * n_pts * sizeof(float);
    int pairwise_dsize = n_clrs * n_pts * sizeof(float);
    int clr_idxs_dsize = n_pts * sizeof(int);
    int clr_sizes_dsize = n_clrs * sizeof(int);
    float one = 1.0, zero = 0.0;

    cudaMalloc((void**)&means_dev, means_dsize);
    cudaMalloc((void**)&points_dev, points_dsize);

    cudaMalloc((void**)&means_sqnorm, n_clrs * sizeof(float));
    cudaMalloc((void**)&points_sqnorm, n_pts * sizeof(float));
    cudaMalloc((void**)&pw_sqdist, pairwise_dsize);

    cudaMalloc((void**)&clr_idxs_dev, clr_idxs_dsize);
    cudaMalloc((void**)&clr_sizes_dev, clr_sizes_dsize);
    cudaMalloc((void**)&avg, pairwise_dsize);

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Initialize kernel config
    dim3 grid(CEIL_DIV(n_clrs, BLOCK_DIM), CEIL_DIV(n_pts, BLOCK_DIM), 1);
    dim3 avg_grid(CEIL_DIV(n_pts, BLOCK_DIM), CEIL_DIV(n_clrs, BLOCK_DIM), 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

    // Load data into device
    cudaMemcpy(means_dev, means, means_dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(points_dev, points, points_dsize, cudaMemcpyHostToDevice);

    for (int it = 0; it < N_ITERS; it++) {
        /* --- Reset buffers --- */
        memset(clr_sizes, 0, clr_sizes_dsize);

        /* --- Compute pairwise square distances --- */
        pairwiseSquaredDist(
            handle, &grid, &block,
            means_dev, points_dev, means_sqnorm, points_sqnorm, pw_sqdist,
            n_clrs, n_pts, n_dims
        );
        cudaDeviceSynchronize();

        /* --- Compute cluster indices and sizes --- */
        argMin<<<CEIL_DIV(n_pts, BLOCK_DIM), BLOCK_DIM>>>(
            pw_sqdist, n_clrs, n_pts, clr_idxs_dev
        );
        cudaDeviceSynchronize();

        cudaMemcpy(clr_idxs, clr_idxs_dev, clr_idxs_dsize, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_pts; i++) {
            clr_sizes[clr_idxs[i]]++;
        }

        /* --- Compute new means --- */
        cudaMemcpy(clr_sizes_dev, clr_sizes, clr_sizes_dsize, cudaMemcpyHostToDevice);
#ifndef GRID1D
        makeAvgMatrix<<<avg_grid, block>>>(avg, clr_idxs_dev, clr_sizes_dev, n_pts, n_clrs);
#else
        const int block_dim = 1024;
        const int grid_dim = CEIL_DIV(n_pts, block_dim);
        makeAvgMatrix<<<grid_dim, block_dim>>>(avg, clr_idxs_dev, clr_sizes_dev, n_pts, n_clrs);
#endif
        cudaDeviceSynchronize();

        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_dims, n_clrs, n_pts,
            &one,
            points_dev, n_dims,
            avg, n_pts,
            &zero,
            means_dev, n_dims
        );
        cudaDeviceSynchronize();

#ifdef DEBUG
        float *pw_sqdist_hst = (float*) malloc(pairwise_dsize);
        cudaMemcpy(pw_sqdist_hst, pw_sqdist, pairwise_dsize, cudaMemcpyDeviceToHost);
        puts("pw_sqdist:");
        print2DArray(pw_sqdist_hst, n_clrs, n_pts);
        free(pw_sqdist_hst);

        puts("clr_sizes:");
        for (int i = 0; i < n_clrs; i++) printf("%d ", clr_sizes[i]);
        puts("\n");

        float *avg_hst = (float*) malloc(pairwise_dsize);
        cudaMemcpy(avg_hst, avg, pairwise_dsize, cudaMemcpyDeviceToHost);
        puts("avg:");
        print2DArray(avg_hst, n_pts, n_clrs);
        free(avg_hst);
#endif
    }
    cudaMemcpy(out_means, means_dev, means_dsize, cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

#ifdef BMK
    printf("%.4f\n", elapsed);
#else
    printf("Time elapsed: %.3f seconds.\n", elapsed);
#endif

    // Free memory
    cudaFree(means_dev);
    cudaFree(points_dev);
    cudaFree(means_sqnorm);
    cudaFree(points_sqnorm);
    cudaFree(pw_sqdist);
    cudaFree(clr_idxs_dev);
    cudaFree(clr_sizes_dev);
    cudaFree(avg);
    cublasDestroy(handle);
}


void pairwiseSquaredDist(
    cublasHandle_t handle, dim3 *grid_p, dim3 *block_p,
    float *q, float *r, float *sqnorm_q, float *sqnorm_r, float *pw_sqdist,
    int n_q, int n_r, int n_d
)
{
    /*
     * Computes pairwise squared distance between 2 sets of vectors.
     * Inputs:
     *   q          : 1st set of vectors.        [n_d, n_q]
     *   r          : 2nd set of vectors.        [n_d, n_r]
     *   pw_sqdist  : Pairwise squared distance. [n_q, n_r]
     *   n_q: Number of vectors in q.
     *   n_r: Number of vectors in r.
     *   n_d: Number of dimensions in each vector.
     */

    float coeff = -2.0, one = 1.0, zero = 0.0;

    // Compute squared norms
    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* The operation executes (A^T) B */
        1, 1, n_d,                /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        &one,                     /* Scalar multiplied before A */
        q, n_d, n_d,              /* Matrix A, its leading dimension, and its stride */
        q, n_d, n_d,              /* Matrix B, its leading dimension, and its stride */
        &zero,                    /* Scalar multiplied before C */
        sqnorm_q, 1, 1,           /* Matrix C, its leading dimension, and its stride */
        n_q                       /* Batch size */
    );

#ifdef DEBUG
    float *sqnorm_q_hst = (float*) malloc(n_q * sizeof(float));
    cudaMemcpy(sqnorm_q_hst, sqnorm_q, n_q * sizeof(float), cudaMemcpyDeviceToHost);
    puts("sqnorm_q:");
    for (int i = 0; i < n_q; i++) printf("%.3f ", sqnorm_q_hst[i]);
    puts("");
    free(sqnorm_q_hst);
#endif

    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* The operation executes (A^T) B */
        1, 1, n_d,                /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        &one,                     /* Scalar multiplied before A */
        r, n_d, n_d,              /* Matrix A, its leading dimension, and its stride */
        r, n_d, n_d,              /* Matrix B, its leading dimension, and its stride */
        &zero,                    /* Scalar multiplied before C */
        sqnorm_r, 1, 1,           /* Matrix C, its leading dimension, and its stride */
        n_r                       /* Batch size */
    );

#ifdef DEBUG
    cudaDeviceSynchronize();
    float *sqnorm_r_hst = (float*) malloc(n_r * sizeof(float));
    cudaMemcpy(sqnorm_r_hst, sqnorm_r, n_r * sizeof(float), cudaMemcpyDeviceToHost);
    puts("sqnorm_r:");
    for (int i = 0; i < n_r; i++) printf("%.3f ", sqnorm_r_hst[i]);
    puts("");
    free(sqnorm_r_hst);
#endif

    // Compute the 3rd term for pairwise squared distance
    cublasSgemm(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* The operation executes (A^T) B */
        n_q, n_r, n_d,            /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        &coeff,                   /* Scalar multiplied before A */
        q, n_d,                   /* Matrix A and its leading dimension */
        r, n_d,                   /* Matrix B and its leading dimension */
        &zero,                    /* Scalar multiplied before C */
        pw_sqdist, n_q            /* Matrix C and its leading dimension */
    );

#ifdef DEBUG
    cudaDeviceSynchronize();
    float *pw_sqdist_hst = (float*) malloc(n_q * n_r * sizeof(float));
    cudaMemcpy(pw_sqdist_hst, pw_sqdist, n_q * n_r * sizeof(float), cudaMemcpyDeviceToHost);
    puts("pw_sqdist:");
    print2DArray(pw_sqdist_hst, n_q, n_r);
    free(pw_sqdist_hst);
#endif

    // Add squared norms to the 3rd term
    addSquaredNorms<<<*grid_p, *block_p>>>(pw_sqdist, sqnorm_q, sqnorm_r, n_q, n_r);
}


__global__ void addSquaredNorms(
    float *pw_sqdist, float *sqnorm_q, float *sqnorm_r, int n_q, int n_r
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < n_q && j < n_r) {
        pw_sqdist[ADDR(i, j, n_q)] += sqnorm_q[i] + sqnorm_r[j];
    }
}


__global__ void argMin(float *arr, int n_rows, int n_cols, int *idxs)
{
    // Reduce along axis 0: [n_rows x n_cols] -> [n_cols]
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < n_cols) {
        float min_val = arr[ADDR(0, j, n_rows)];
        int arg_min = 0;

        for (int i = 1; i < n_rows; i++) {
            if (arr[ADDR(i, j, n_rows)] < min_val) {
                min_val = arr[ADDR(i, j, n_rows)];
                arg_min = i;
            }
        }

        idxs[j] = arg_min;
    }
}


__global__ void makeAvgMatrix(float *avg, int *clr_idxs, int *clr_sizes, int n_pts, int n_clrs)
{
#ifndef GRID1D
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_pts && j < n_clrs) {
        if (j == clr_idxs[i]) {
            avg[ADDR(i, j, n_pts)] = 1.0 / clr_sizes[j];
        }
        else {
            avg[ADDR(i, j, n_pts)] = 0.0;
        }
    }
#else
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n_pts) {
        for (int j = 0; j < n_clrs; j++) {
            avg[ADDR(i, j, n_pts)] = 0.0;
        }
        avg[ADDR(i, clr_idxs[i], n_pts)] = 1.0 / clr_sizes[clr_idxs[i]];
    }
#endif
}
