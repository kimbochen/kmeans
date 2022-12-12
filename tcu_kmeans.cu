#include <stdio.h>  // puts
#include <time.h>   // clock

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "kmeans.h"


void tcuKMeans(
    float *means, float *points, int *clr_sizes, int *clr_idxs,
    int n_clrs, int n_pts, int n_dims, float *out_means
)
{
#ifndef BMK
    puts("----- Benchmarking Tensor Core version -----");
#endif

    clock_t start = clock();

    // Arrays
    // __half *means_hf, *points_hf;
    float *means_sg, *points_sg;
    int *clr_idxs_dev, *clr_sizes_dev;
    __half *means_dev, *points_dev;
    __half *means_sqnorm, *points_sqnorm, *pw_sqdist;
    __half *avg;

    // Constants
    int means_dsize = n_dims * n_clrs * sizeof(__half);
    int points_dsize = n_dims * n_pts * sizeof(__half);
    int pairwise_dsize = n_clrs * n_pts * sizeof(__half);
    int clr_idxs_dsize = n_pts * sizeof(int);
    int clr_sizes_dsize = n_clrs * sizeof(int);
    __half one = 1.0, zero = 0.0;

    // means_hf = (__half*) malloc(means_dsize);
    // points_hf = (__half*) malloc(points_dsize);
    cudaMalloc((void**)&means_sg, means_dsize * 2);
    cudaMalloc((void**)&points_sg, points_dsize * 2);

    cudaMalloc((void**)&means_dev, means_dsize);
    cudaMalloc((void**)&points_dev, points_dsize);

    cudaMalloc((void**)&means_sqnorm, n_clrs * sizeof(__half));
    cudaMalloc((void**)&points_sqnorm, n_pts * sizeof(__half));
    cudaMalloc((void**)&pw_sqdist, pairwise_dsize);

    cudaMalloc((void**)&clr_idxs_dev, clr_idxs_dsize);
    cudaMalloc((void**)&clr_sizes_dev, clr_sizes_dsize);
    cudaMalloc((void**)&avg, pairwise_dsize);

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    // Initialize kernel config
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid(CEIL_DIV(n_clrs, BLOCK_DIM), CEIL_DIV(n_pts, BLOCK_DIM));
    dim3 avg_grid(CEIL_DIV(n_pts, BLOCK_DIM), CEIL_DIV(n_clrs, BLOCK_DIM), 1);

    // Load data into device
    cudaMemcpy(means_sg, means, means_dsize * 2, cudaMemcpyHostToDevice);
    float2Half<<<CEIL_DIV(n_dims * n_clrs, 1024), 1024>>>(means_sg, means_dev, n_dims * n_clrs);

    cudaMemcpy(points_sg, points, points_dsize * 2, cudaMemcpyHostToDevice);
    float2Half<<<CEIL_DIV(n_dims * n_pts, 1024), 1024>>>(points_sg, points_dev, n_dims * n_pts);
    cudaDeviceSynchronize();

    // for (int i = 0; i < n_dims * n_clrs; i++) {
    //     means_hf[i] = __float2half(means[i]);
    // }
    // cudaMemcpy(means_dev, means_hf, means_dsize, cudaMemcpyHostToDevice);

    // for (int i = 0; i < n_dims * n_pts; i++) {
    //     points_hf[i] = __float2half(points[i]);
    // }
    // cudaMemcpy(points_dev, points_hf, points_dsize, cudaMemcpyHostToDevice);

    // Start KMeans
    for (int it = 0; it < N_ITERS; it++) {
        /* --- Reset buffers --- */
        memset(clr_sizes, 0, clr_sizes_dsize);

        /* --- Compute pairwise square distances --- */
        pairwiseSquaredDistHalf(
            handle, &grid, &block,
            means_dev, points_dev, means_sqnorm, points_sqnorm, pw_sqdist,
            n_clrs, n_pts, n_dims
        );
        cudaDeviceSynchronize();

        /* --- Compute cluster indices and sizes --- */
        argMinHalf<<<CEIL_DIV(n_pts, BLOCK_DIM), BLOCK_DIM>>>(
            pw_sqdist, n_clrs, n_pts, clr_idxs_dev
        );
        cudaDeviceSynchronize();

        cudaMemcpy(clr_idxs, clr_idxs_dev, clr_idxs_dsize, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n_pts; i++) {
            clr_sizes[clr_idxs[i]]++;
        }

        /* --- Compute new means --- */
        cudaMemcpy(clr_sizes_dev, clr_sizes, clr_sizes_dsize, cudaMemcpyHostToDevice);
        makeAvgMatrixHalf<<<avg_grid, block>>>(avg, clr_idxs_dev, clr_sizes_dev, n_pts, n_clrs);
        cudaDeviceSynchronize();

#ifdef DEBUG
        __half *points_hst = (__half*) malloc(points_dsize);
        cudaMemcpy(points_hst, points_dev, points_dsize, cudaMemcpyDeviceToHost);
        puts("points_hst:");
        for (int i = 0; i < n_dims; i++) {
            for (int j = 0; j < n_pts; j++) {
                printf("%.3f ", __half2float(points_hst[ADDR(i, j, n_dims)]));
            }
            puts("");
        }
        puts("");
        free(points_hst);

        __half *avg_hst = (__half*) malloc(pairwise_dsize);
        cudaMemcpy(avg_hst, avg, pairwise_dsize, cudaMemcpyDeviceToHost);
        puts("avg:");
        for (int i = 0; i < n_pts; i++) {
            for (int j = 0; j < n_clrs; j++)
                printf("%.3f ", __half2float(avg_hst[ADDR(i, j, n_pts)]));
            puts("");
        }
        puts("");
        free(avg_hst);

        __half *means_hst = (__half*) malloc(means_dsize);
        cudaMemcpy(means_hst, means_dev, means_dsize, cudaMemcpyDeviceToHost);
        puts("means_hst:");
        for (int i = 0; i < n_dims; i++) {
            for (int j = 0; j < n_clrs; j++) {
                printf("%.3f ", __half2float(means_hst[ADDR(i, j, n_dims)]));
            }
            puts("");
        }
        puts("");
        free(means_hst);
#endif

        cublasStatus_t status = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_dims, n_clrs, n_pts,
            &one,  points_dev, CUDA_R_16F, n_dims,
                   avg       , CUDA_R_16F, n_pts,
            &zero, means_dev , CUDA_R_16F, n_dims,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        cudaDeviceSynchronize();

#ifdef DEBUG
        printf("CUBLAS_STATUS_SUCCESS: %d\n", CUBLAS_STATUS_SUCCESS);
        printf("CUBLAS_STATUS_NOT_INITIALIZED: %d\n", CUBLAS_STATUS_NOT_INITIALIZED);
        printf("CUBLAS_STATUS_ARCH_MISMATCH: %d\n", CUBLAS_STATUS_ARCH_MISMATCH);
        printf("CUBLAS_STATUS_NOT_SUPPORTED: %d\n", CUBLAS_STATUS_NOT_SUPPORTED);
        printf("CUBLAS_STATUS_INVALID_VALUE: %d\n", CUBLAS_STATUS_INVALID_VALUE);
        printf("CUBLAS_STATUS_EXECUTION_FAILED: %d\n", CUBLAS_STATUS_EXECUTION_FAILED);
        printf("status: %d\n\n", status);
#endif

#ifdef DEBUG
        __half *pw_sqdist_hst = (__half*) malloc(pairwise_dsize);
        cudaMemcpy(pw_sqdist_hst, pw_sqdist, pairwise_dsize, cudaMemcpyDeviceToHost);
        puts("pw_sqdist:");
        for (int i = 0; i < n_clrs; i++) {
            for (int j = 0; j < n_pts; j++)
                printf("%.3f ", __half2float(pw_sqdist_hst[ADDR(i, j, n_clrs)]));
            puts("");
        }
        puts("");
        free(pw_sqdist_hst);

        puts("clr_sizes:");
        for (int i = 0; i < n_clrs; i++) printf("%d ", clr_sizes[i]);
        puts("\n");

        __half *means_hst2 = (__half*) malloc(means_dsize);
        cudaMemcpy(means_hst2, means_dev, means_dsize, cudaMemcpyDeviceToHost);
        puts("means:");
        for (int i = 0; i < n_dims; i++) {
            for (int j = 0; j < n_clrs; j++) {
                printf("%.3f ", __half2float(means_hst2[ADDR(i, j, n_dims)]));
            }
            puts("");
        }
        puts("");
        free(means_hst2);
#endif
    }
    // cudaMemcpy(means_hf, means_dev, means_dsize, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n_dims * n_clrs; i++) {
    //     out_means[i] = __half2float(means_hf[i]);
    // }
    half2Float<<<CEIL_DIV(n_dims * n_clrs, 1024), 1024>>>(means_dev, means_sg, n_dims * n_clrs);
    cudaDeviceSynchronize();
    cudaMemcpy(out_means, means_sg, means_dsize * 2, cudaMemcpyDeviceToHost);


    clock_t end = clock();
    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

#ifdef BMK
    printf("%.4f\n", elapsed);
#else
    printf("Time elapsed: %.3f seconds.\n", elapsed);
#endif

    // Free memory
    // free(means_hf);
    // free(points_hf);
    cudaFree(means_sg);
    cudaFree(points_sg);
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


__global__ void float2Half(float *arr_sg, __half *arr_hf, int n_elems)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < n_elems) {
        arr_hf[tid] = __float2half(arr_sg[tid]);
    }
}


__global__ void half2Float(__half *arr_hf, float *arr_sg, int n_elems)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < n_elems) {
        arr_sg[tid] = __half2float(arr_hf[tid]);
    }
}


void pairwiseSquaredDistHalf(
    cublasHandle_t handle, dim3 *grid_p, dim3 *block_p,
    __half *q, __half *r, __half *sqnorm_q, __half *sqnorm_r, __half *pw_sqdist,
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

    __half coeff = -2.0, one = 1.0, zero = 0.0;

    // Compute squared norms
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        1, 1, n_d,
        &one,  q,        CUDA_R_16F, n_d, n_d,
               q,        CUDA_R_16F, n_d, n_d,
        &zero, sqnorm_q, CUDA_R_16F, 1,   1,
        n_q,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT
    );

#ifdef DEBUG
    __half *sqnorm_q_hst = (__half*) malloc(n_q * sizeof(__half));
    cudaMemcpy(sqnorm_q_hst, sqnorm_q, n_q * sizeof(__half), cudaMemcpyDeviceToHost);
    puts("sqnorm_q:");
    for (int i = 0; i < n_q; i++) printf("%.3f ", __half2float(sqnorm_q_hst[i]));
    puts("\n");
    free(sqnorm_q_hst);
#endif


    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        1, 1, n_d,
        &one,  r,        CUDA_R_16F, n_d, n_d,
               r,        CUDA_R_16F, n_d, n_d,
        &zero, sqnorm_r, CUDA_R_16F, 1,   1,
        n_r,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT
    );
    cudaDeviceSynchronize();

#ifdef DEBUG
    __half *sqnorm_r_hst = (__half*) malloc(n_r * sizeof(__half));
    cudaMemcpy(sqnorm_r_hst, sqnorm_r, n_r * sizeof(__half), cudaMemcpyDeviceToHost);
    puts("sqnorm_r:");
    for (int i = 0; i < n_r; i++) printf("%.3f ", __half2float(sqnorm_r_hst[i]));
    puts("\n");
    free(sqnorm_r_hst);
#endif

    // Compute the 3rd term for pairwise squared distance
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n_q, n_r, n_d,
        &coeff, q,         CUDA_R_16F, n_d,
                r,         CUDA_R_16F, n_d,
        &zero,  pw_sqdist, CUDA_R_16F, n_q,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cudaDeviceSynchronize();

#ifdef DEBUG
    __half *pw_sqdist_hst = (__half*) malloc(n_q * n_r * sizeof(__half));
    cudaMemcpy(pw_sqdist_hst, pw_sqdist, n_q * n_r * sizeof(__half), cudaMemcpyDeviceToHost);
    puts("pw_sqdist (3rd term):");
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < n_r; j++)
            printf("%.3f ", __half2float(pw_sqdist_hst[ADDR(i, j, n_q)]));
        puts("");
    }
    puts("");
    free(pw_sqdist_hst);
#endif


    // Add squared norms to the 3rd term
    addSquaredNormsHalf<<<*grid_p, *block_p>>>(pw_sqdist, sqnorm_q, sqnorm_r, n_q, n_r);
}


__global__ void addSquaredNormsHalf(
    __half *pw_sqdist, __half *sqnorm_q, __half *sqnorm_r, int n_q, int n_r
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_q && j < n_r) {
        __half sqnorms = __hadd(sqnorm_q[i], sqnorm_r[j]);
        pw_sqdist[ADDR(i, j, n_q)] = __hadd(pw_sqdist[ADDR(i, j, n_q)], sqnorms);
    }
}


__global__ void argMinHalf(__half *arr, int n_rows, int n_cols, int *idxs)
{
    // Reduce along axis 0: [n_rows x n_cols] -> [n_cols]
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < n_cols) {
        __half min_val = arr[ADDR(0, j, n_rows)];
        int arg_min = 0;

        for (int i = 1; i < n_rows; i++) {
            if (__hlt(arr[ADDR(i, j, n_rows)], min_val)) {
                min_val = arr[ADDR(i, j, n_rows)];
                arg_min = i;
            }
        }

        idxs[j] = arg_min;
    }
}


__global__ void makeAvgMatrixHalf(__half *avg, int *clr_idxs, int *clr_sizes, int n_pts, int n_clrs)
{
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
}
