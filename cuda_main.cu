#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_kmeans.h"
#include "file_io.h"


int main(int argc, char **argv)
{
    assert(argc == 3);

    int n_pts, n_clusters, n_dims;
    float *points, *means;
    int *membership, *cluster_sizes;

    readPoints1D(argv[1], &points, &n_pts, &n_dims);
    initMeans1D(argv[2], &means, &n_clusters, points, n_pts, n_dims);

    membership = (int*) malloc(n_pts * sizeof(int));
    for (int i = 0; i < n_pts; i++) {
        membership[i] = -1;
    }

    cluster_sizes = (int*) malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++) {
        cluster_sizes[i] = 0;
    }

    cudaKMeans(means, points, cluster_sizes, membership, n_clusters, n_pts, n_dims);

    free(points);
    free(means);
    free(membership);
    free(cluster_sizes);
}
