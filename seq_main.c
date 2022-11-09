#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "file_io.h"
#include "seq_kmeans.h"


int main(int argc, char **argv)
{
    assert(argc == 3);

    int n_pts, n_dims, n_clusters;
    float **points, **means;

    readPoints(argv[1], &points, &n_pts, &n_dims);
    initMeans(argv[2], &means, &n_clusters, points, n_pts, n_dims);

    int *cluster_sizes = (int*) malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++) {
        cluster_sizes[i] = 0;
    }

    int *membership = (int*) malloc(n_pts * sizeof(int));
    for (int i = 0; i < n_pts; i++) {
        membership[i] = -1;
    }

    seqKMeans(points, means, cluster_sizes, membership, n_pts, n_dims, n_clusters);

#ifdef PRINT
    printf("# %d %d\n", n_clusters, n_dims);
    print2DFloatArray(means, n_clusters, n_dims);
#endif

    free2DFloatArray(points);
    free2DFloatArray(means);
    free(cluster_sizes);
    free(membership);
}
