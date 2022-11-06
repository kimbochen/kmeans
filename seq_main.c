#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "file_io.h"
#include "seq_kmeans.h"


int main(int argc, char **argv)
{
    assert(argc == 3);

    int n_pts, n_feats, n_clusters;
    float **points, **means;

    readPoints(argv[1], &points, &n_pts, &n_feats);
    initMeans(argv[2], &means, &n_clusters, points, n_pts, n_feats);

    float **clusters = malloc2DFloatArray(n_clusters, n_feats);
    for (int i = 0; i < n_clusters * n_feats; i++) {
        clusters[0][i] = 0.0;
    }

    int *cluster_sizes = (int*) malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++) {
        cluster_sizes[i] = 0;
    }

    seqKMeans(points, means, clusters, cluster_sizes, n_pts, n_feats, n_clusters);

#ifdef PRINT
    printf("# %d %d\n", n_clusters, n_feats);
    print2DFloatArray(means, n_clusters, n_feats);
#endif

    free2DFloatArray(points);
    free2DFloatArray(means);
    free2DFloatArray(clusters);
    free(cluster_sizes);
}
