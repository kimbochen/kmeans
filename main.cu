#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"


int main(int argc, char **argv)
{
    /*
     * Prepares the data
     * Executes the routine
     * Collects the timing
     * Prints the results
     */
    assert(argc == 4);

    int n_pts = atoi(argv[1]);
    int n_dims = atoi(argv[2]);
    int n_clrs = atoi(argv[3]);


    float *points = (float*) malloc(n_dims * n_pts * sizeof(float));
    float *means = (float*) malloc(n_dims * n_clrs * sizeof(float));
    float *out_means = (float*) malloc(n_dims * n_clrs * sizeof(float));
    int *clr_idxs = (int*) malloc(n_pts * sizeof(int));
    int *clr_sizes = (int*) malloc(n_clrs * sizeof(int));

    srand(3985);
    for (int i = 0; i < n_dims; i++) {
        for (int j = 0; j < n_pts; j++) {
            points[ADDR(i, j, n_dims)] = (rand() % 1000) * 0.01;
        }
    }
    for (int i = 0; i < n_dims; i++) {
        for (int j = 0; j < n_clrs; j++) {
            means[ADDR(i, j, n_dims)] = points[ADDR(i, j, n_dims)];
            out_means[ADDR(i, j, n_dims)] = points[ADDR(i, j, n_dims)];
        }
    }
    for (int i = 0; i < n_pts; i++) {
        clr_idxs[i] = -1;
    }

#ifdef BMK
    printf("%d,%d,%d,", n_pts, n_dims, n_clrs);
#endif

#if KERNEL == 1
    cudaKMeans(means, points, clr_sizes, clr_idxs, n_clrs, n_pts, n_dims, out_means);
    // computeInertia(out_means, points, clr_idxs, n_clrs, n_pts, n_dims);
#elif KERNEL == 2
    cublasKMeans(means, points, clr_sizes, clr_idxs, n_clrs, n_pts, n_dims, out_means);
    // computeInertia(out_means, points, clr_idxs, n_clrs, n_pts, n_dims);
#elif KERNEL == 3
    tcuKMeans(means, points, clr_sizes, clr_idxs, n_clrs, n_pts, n_dims, out_means);
    // computeInertia(out_means, points, clr_idxs, n_clrs, n_pts, n_dims);
#else
    puts("No kernel is selected.");
#endif

    free(points);
    free(means);
    free(clr_idxs);
    free(clr_sizes);
    free(out_means);
}


void computeInertia(float *means, float *points, int *clr_idxs, int n_clrs, int n_pts, int n_dims)
{
    float inertia = 0.0;

    for (int j = 0; j < n_pts; j++) {
        int clr_idx = clr_idxs[j];

        for (int i = 0; i < n_dims; i++) {
            float diff = points[ADDR(i, j, n_dims)] - means[ADDR(i, clr_idx, n_dims)];
            inertia += diff * diff;
        }
    }

#ifdef DEBUG
    puts("means:");
    print2DArray(means, n_dims, n_clrs);

    puts("clr_idxs:");
    for (int i = 0; i < n_pts; i++) printf("%d ", clr_idxs[i]);
    puts("");
#endif
    printf("Inertia: %.4f\n\n", inertia);
}


void print2DArray(float *arr, int dim0, int dim1)
{
    printf("dim: [%d x %d]\n", dim0, dim1);
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++)
            printf("%.3f ", arr[ADDR(i, j, dim0)]);
        puts("");
    }
    puts("");
}
