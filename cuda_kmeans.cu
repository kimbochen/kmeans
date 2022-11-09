#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include "cuda_kmeans.h"

#define TOL 1e-4
#define MAX_ITER 300


void cudaKMeans(
    float *means, float *points,
    int *cluster_sizes, int *membership,
    int n_clusters, int n_pts, int n_dims
)
{
    clock_t start = clock();

    // Allocate memory
    float *means_dev, *points_dev;
    float *means_sqnorm, *points_sqnorm, *pw_sqdist_dev;
    float coeff = -2.0, one = 1.0, zero = 0.0;
    float delta;

    int *membership_dev, *cluster_sizes_dev;
    int *new_membership = (int*) malloc(n_pts * sizeof(int));
    float *avg_dev;

    cudaMalloc((void**)&means_dev, n_dims * n_clusters * sizeof(float));
    cudaMalloc((void**)&points_dev, n_dims * n_pts * sizeof(float));

    cudaMalloc((void**)&means_sqnorm, n_clusters * sizeof(float));
    cudaMalloc((void**)&points_sqnorm, n_pts * sizeof(float));
    cudaMalloc((void**)&pw_sqdist_dev, n_clusters * n_pts * sizeof(float));

    cudaMalloc((void**)&membership_dev, n_pts * sizeof(int));
    cudaMalloc((void**)&cluster_sizes_dev, n_pts * sizeof(int));
    cudaMalloc((void**)&avg_dev, n_pts * n_clusters * sizeof(float));

    // Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    // Initialize kernel config
    dim3 grid(CEIL_DIV(n_clusters, BLOCK_DIM), CEIL_DIV(n_pts, BLOCK_DIM), 1);
    dim3 grid1(CEIL_DIV(n_pts, BLOCK_DIM), CEIL_DIV(n_clusters, BLOCK_DIM), 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

    // Load data into device
    cudaMemcpy(means_dev, means, n_dims * n_clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(points_dev, points, n_dims * n_pts * sizeof(float), cudaMemcpyHostToDevice);

    int it;
    for (it = 0; it < MAX_ITER; it++) {
        // Calculate new assignments
        pairwiseSquaredDist(
            handle,
            means_dev, points_dev,
            means_sqnorm, points_sqnorm,
            n_clusters, n_pts, n_dims,
            &coeff, &zero, &one,
            &grid, &block,
            pw_sqdist_dev
        );
        argMin<<<n_pts, n_clusters>>>(pw_sqdist_dev, n_clusters, n_pts, membership_dev);
        cudaDeviceSynchronize();
        cudaMemcpy(new_membership, membership_dev, n_pts * sizeof(int), cudaMemcpyDeviceToHost);

        // Check convergence
        delta = 0.0;
        for (int i = 0; i < n_pts; i++) {
            if (new_membership[i] != membership[i]) {
                membership[i] = new_membership[i];
                delta += 1.0;
            }
        }
        if (delta / n_pts < TOL) {
            break;
        }

        // Update means
        for (int i = 0; i < n_pts; i++) {
            cluster_sizes[membership[i]]++;
        }
        cudaMemcpy(cluster_sizes_dev, cluster_sizes, n_pts * sizeof(int), cudaMemcpyHostToDevice);
        makeAvgMatrix<<<grid1, block>>>(avg_dev, membership_dev, cluster_sizes_dev, n_pts, n_clusters);
        cudaDeviceSynchronize();
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n_dims, n_clusters, n_pts,
            &one,
            points_dev, n_dims,
            avg_dev, n_pts,
            &zero,
            means_dev, n_dims
        );
        cudaDeviceSynchronize();

        // Reset buffer
        for (int i = 0; i < n_clusters; i++) {
            cluster_sizes[i] = 0;
        }
    }
    cudaMemcpy(means, means_dev, n_dims * n_clusters * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

    float inertia = 0.0;
    for (int i = 0; i < n_pts; i++) {
        int m_idx = membership[i];
        for (int d = 0; d < n_dims; d++) {
            float diff = points[d + i * n_dims] - means[d + m_idx * n_dims];
            inertia += diff * diff;
        }
    }
    printf("Converged in %d iterations. Inertia = %.3f. Elapsed = %.3f sec\n", it, inertia, elapsed);

    // Free memory
    cudaFree(means_dev);
    cudaFree(points_dev);
    cudaFree(means_sqnorm);
    cudaFree(points_sqnorm);
    cudaFree(pw_sqdist_dev);
    cudaFree(membership_dev);
    cudaFree(cluster_sizes_dev);
    cudaFree(avg_dev);
    free(new_membership);
    cublasDestroy(handle);
}


void pairwiseSquaredDist(
    cublasHandle_t handle,
    float *q, float *r,
    float *sqnorm_q, float *sqnorm_r,
    int n_q, int n_r, int n_d,
    float *coeff_p, float *zero_p, float *one_p,
    dim3 *grid_p, dim3 *block_p,
    float *pw_sqdist
)
{
    /*
     * Computes pairwise squared distance between 2 sets of vectors.
     * Inputs:
     *   q          : 1st set of vectors.        [n_d, n_q]
     *   r          : 2nd set of vectors.        [n_d, n_r]
     *   pw_sqdist: Pairwise squared distance. [n_q, n_r]
     *   n_q: Number of vectors in q.
     *   n_r: Number of vectors in r.
     *   n_d: Number of dimensions in each vector.
     */

    // Compute squared norms
    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        1, 1, n_d,              /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        one_p,                    /* Scalar multiplied before A */
        q, n_d, n_d,             /* Matrix A, its leading dimension, and its stride */
        q, n_d, n_d,             /* Matrix B, its leading dimension, and its stride */
        zero_p,                   /* Scalar multiplied before C */
        sqnorm_q, 1, 1,         /* Matrix C, its leading dimension, and its stride */
        n_q
    );

    cublasSgemmStridedBatched(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        1, 1, n_d,              /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        one_p,                    /* Scalar multiplied before A */
        r, n_d, n_d,             /* Matrix A, its leading dimension, and its stride */
        r, n_d, n_d,             /* Matrix B, its leading dimension, and its stride */
        zero_p,                   /* Scalar multiplied before C */
        sqnorm_r, 1, 1,         /* Matrix C, its leading dimension, and its stride */
        n_r
    );
    cudaDeviceSynchronize();

    // Compute the 3rd term for pairwise squared distance
    cublasSgemm(
        handle,                   /* cuBLAS handle */
        CUBLAS_OP_T, CUBLAS_OP_N, /* Transpose status for A and B */
        n_q, n_r, n_d,            /* Matrix dimensions m, n, k: A(m x k), B(k x n) */
        coeff_p,                  /* Scalar multiplied before A */
        q, n_d,                   /* Matrix A and its leading dimension */
        r, n_d,                   /* Matrix B and its leading dimension */
        zero_p,                   /* Scalar multiplied before C */
        pw_sqdist, n_q          /* Matrix C and its leading dimension */
    );
    cudaDeviceSynchronize();

    // Add squared norms to the 3rd term
    addSquaredNorms<<<*grid_p, *block_p>>>(pw_sqdist, sqnorm_q, sqnorm_r, n_q, n_r);
    cudaDeviceSynchronize();
}


__global__ void addSquaredNorms(
    float *pw_sqdist, float *sqnorm_q, float *sqnorm_r, int n_q, int n_r
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_q && j < n_r) {
        pw_sqdist[i + j * n_q] += sqnorm_q[i] + sqnorm_r[j];
    }
}


__global__ void argMin(float *arr, int n_rows, int n_cols, int *idxs)
{
    // Reduce along axis 0: [n_rows x n_cols] -> [n_cols]
    int tid = threadIdx.x;
    int j = blockIdx.x;

    if (tid == 0) {
        int min_val = INT_MAX;

        for (int i = 0; i < n_rows; i++) {
            if (arr[i + j * n_rows] < min_val) {
                idxs[j] = i;
                min_val = arr[i + j * n_rows];
            }
        }
    }
}


__global__ void makeAvgMatrix(float *avg, int *membership, int *cluster_sizes, int n_r, int n_q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_r && j < n_q) {
        if (j == membership[i]) {
            avg[i + j * n_r] = 1.0 / cluster_sizes[j];
        }
        else {
            avg[i + j * n_r] = 0.0;
        }
    }
}


void readPoints1D(char *fname, float **points_p, int *n_pts_p, int *n_dims_p)
{
    FILE *fin = fopen(fname, "r");
    assert(fin != NULL);

    int status = fscanf(fin, "# %d %d", n_pts_p, n_dims_p);
    assert(status == 2);

    int n_pts = *n_pts_p;
    int n_dims = *n_dims_p;

    float *points = (float*) malloc(n_pts * n_dims * sizeof(float));
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < n_dims; j++) {
            int s = fscanf(fin, "%f", &points[j + i * n_dims]);
            assert(s == 1);
        }
    }

    fclose(fin);

    *points_p = points;
}


void initMeans1D(char *fname, float **means_p, int *n_clusters_p, float *points, int n_pts, int n_dims)
{
    FILE *fin = fopen(fname, "r");
    assert(fin != NULL);

    int status = fscanf(fin, "# %d", n_clusters_p);
    assert(status == 1);

    int idx;
    int n_clusters = *n_clusters_p;
    float *means = (float*) malloc(n_clusters * n_dims * sizeof(float));

    for (int i = 0; i < n_clusters; i++) {
        int s = fscanf(fin, "%d", &idx);
        assert(s == 1);

        for (int j = 0; j < n_dims; j++) {
            means[j + i * n_dims] = points[j + idx * n_dims];
        }
    }

    fclose(fin);

    *means_p = means;
}


// int main(void)
// {
// #define D 2
// #define N 3
// #define M 15
//     float Q[D * N] = {
//         /*
//         80, 70,
//         60, 70,
//         80, 90
//         */
//         78, 68,
//         81, 71,
//         82, 72
//     };
//     float R[D * M] = {
//         80, 70, 60, 70, 80, 90, 79, 69, 59, 69, 79, 89, 81, 71, 61, 71, 81,
//         91, 78, 68, 58, 68, 78, 88, 82, 72, 62, 72, 82, 92
//     };
//     // float pw_sqdist[N * M];
//     int membership[M];
//     int cluster_sizes[N];
// 
//     puts("Q:");
//     for (int i = 0; i < D; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%.3f ", Q[i + j * D]);  // Stored in column major
//         }
//         puts("");
//     }
//     puts("");
// 
//     puts("R:");
//     for (int i = 0; i < D; i++) {
//         for (int j = 0; j < M; j++) {
//             printf("%.3f ", R[i + j * D]);  // Stored in column major
//         }
//         puts("");
//     }
//     puts("");
// 
//     for (int i = 0; i < M; i++) {
//         membership[i] = -1;
//     }
//     for (int i = 0; i < N; i++) {
//         cluster_sizes[i] = 0;
//     }
// 
//     cudaKMeans(Q, R, cluster_sizes, membership, N, M, D);
// 
//     puts("new Q:");
//     for (int i = 0; i < D; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%.3f ", Q[i + j * D]);  // Stored in column major
//         }
//         puts("");
//     }
//     puts("");

    /*
    float *Q_dev, *R_dev;
    int Q_data_size = D * N * sizeof(float);
    int R_data_size = D * M * sizeof(float);

    cudaMalloc((void**)&Q_dev, Q_data_size);
    cudaMemcpy(Q_dev, Q, Q_data_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&R_dev, R_data_size);
    cudaMemcpy(R_dev, R, R_data_size, cudaMemcpyHostToDevice);


    float *Q_sqnorm, *R_sqnorm;
    float *pw_sqdist_dev;
    float coeff = -2.0, one = 1.0, zero = 0.0;

    cudaMalloc((void**)&Q_sqnorm, N * sizeof(float));
    cudaMalloc((void**)&R_sqnorm, M * sizeof(float));
    cudaMalloc((void**)&pw_sqdist_dev, N * M * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

    dim3 grid(CEIL_DIV(N, BLOCK_DIM), CEIL_DIV(M, BLOCK_DIM), 1);
    dim3 block(BLOCK_DIM, BLOCK_DIM, 1);

    pairwiseSquaredDist(
        handle,
        Q_dev, R_dev,
        Q_sqnorm, R_sqnorm,
        N, M, D,
        &coeff, &zero, &one,
        &grid, &block,
        pw_sqdist_dev
    );
    cudaMemcpy(pw_sqdist, pw_sqdist_dev, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    puts("pw_sqdist:");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            printf("%.3f ", pw_sqdist[i + j * N]);  // Stored in column major
        }
        puts("");
    }
    puts("");

    int membership[M];
    int *membership_dev;

    puts("membership:");
    for (int i = 0; i < M; i++) {
        membership[i] = -1;
        printf("%d ", membership[i]);
    }
    puts("");

    cudaMalloc((void**)&membership_dev, M * sizeof(int));

    argMin<<<M, N>>>(pw_sqdist_dev, N, M, membership_dev);

    cudaMemcpy(membership, membership_dev, M * sizeof(int), cudaMemcpyDeviceToHost);

    puts("new membership:");
    for (int i = 0; i < M; i++) {
        printf("%d ", membership[i]);
    }
    puts("");


    int cluster_sizes[M];
    int *cluster_sizes_dev;

    for (int i = 0; i < M; i++) {
        cluster_sizes[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        cluster_sizes[membership[i]]++;
    }
    cudaMalloc((void**)&cluster_sizes_dev, M * sizeof(int));
    cudaMemcpy(cluster_sizes_dev, cluster_sizes, M * sizeof(int), cudaMemcpyHostToDevice);

    float avg[M * N];
    float *avg_dev;

    cudaMalloc((void**)&avg_dev, M * N * sizeof(float));

    dim3 grid1(CEIL_DIV(M, BLOCK_DIM), CEIL_DIV(N, BLOCK_DIM), 1);
    dim3 block1(BLOCK_DIM, BLOCK_DIM, 1);
    makeAvgMatrix<<<grid1, block1>>>(avg_dev, membership_dev, cluster_sizes_dev, M, N);

    cudaMemcpy(avg, avg_dev, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    puts("avg:");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.3f ", avg[i + j * M]);
        }
        puts("");
    }
    puts("");

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        D, N, M,
        &one,
        R_dev, D,
        avg_dev, M,
        &zero,
        Q_dev, D
    );
    cudaMemcpy(Q, Q_dev, Q_data_size, cudaMemcpyDeviceToHost);
    puts("new Q:");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.3f ", Q[i + j * D]);  // Stored in column major
        }
        puts("");
    }
    puts("");


    cudaFree(Q_dev);
    cudaFree(R_dev);
    cudaFree(Q_sqnorm);
    cudaFree(R_sqnorm);
    cudaFree(pw_sqdist_dev);
    cudaFree(membership_dev);
    cudaFree(cluster_sizes_dev);
    cudaFree(avg_dev);
    cublasDestroy(handle);
    */
// }
