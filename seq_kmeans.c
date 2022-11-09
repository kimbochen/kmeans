#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#include "file_io.h"
#include "seq_kmeans.h"

#define TOL 1e-4
#define MAX_ITER 300


void seqKMeans(
    float **points, float **means, int *cluster_sizes, int *membership,
    int n_pts, int n_dims, int n_clusters
)
{
    /*
     * points  : Data points.                                [n_pts, n_dims]
     * means   : Centroids (means) of the data points.       [n_clusters, n_dims]
     * clusters: Buffer for computing new centroids (means). [n_clusters, n_dims]
     * cluster_sizes: Size of each new cluster.              [n_clusters]
     */

    clock_t start = clock();

    float **clusters = malloc2DFloatArray(n_clusters, n_dims);
    for (int i = 0; i < n_clusters * n_dims; i++) {
        clusters[0][i] = 0.0;
    }

    int it;
    float delta = FLT_MAX;

    // Iterate while not exceeding maximum no. of iterations and delta < tolerance
    for (it = 0; it < MAX_ITER; it++) {
        // For every data point: find its nearest mean and add it to the corresponding cluster
        delta = 0.0;

        for (int i = 0; i < n_pts; i++) {
            int m_idx = findNearestMean(means, points[i], n_dims, n_clusters);
            if (m_idx != membership[i]) {
                membership[i] = m_idx;
                delta += 1.0;
            }
            cluster_sizes[m_idx]++;
            for (int d = 0; d < n_dims; d++) {
                clusters[m_idx][d] += points[i][d];
            }
        }

        if (delta / n_pts < TOL) {
            break;
        }

        // Compute the new means
        for (int i = 0; i < n_clusters; i++) {
            for (int d = 0; d < n_dims; d++) {
                clusters[i][d] /= cluster_sizes[i];
            }
        }

        // Update the new means
        for (int i = 0; i < n_clusters; i++) {
            for (int d = 0; d < n_dims; d++) {
                means[i][d] = clusters[i][d];
            }
        }

        // Reset buffers
        for (int i = 0; i < n_clusters * n_dims; i++) {
            clusters[0][i] = 0.0;
        }
        for (int i = 0; i < n_clusters; i++) {
            cluster_sizes[i] = 0;
        }
    }

    clock_t end = clock();
    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

    float inertia = 0.0;
    for (int i = 0; i < n_pts; i++) {
        int m_idx = findNearestMean(means, points[i], n_dims, n_clusters);
        inertia += sumOfSquare(means[m_idx], points[i], n_dims);
    }

    printf("Converged in %d iterations. Inertia = %.3f. Elapsed = %.3f sec\n", it, inertia, elapsed);

    free2DFloatArray(clusters);
}


float sumOfSquare(float *x, float *y, int n_dims)
{
    float sosq = 0.0;

    for (int d = 0; d < n_dims; d++) {
        sosq += (x[d] - y[d]) * (x[d] - y[d]);
    }

    return sosq;
}


int findNearestMean(float **means, float *point, int n_dims, int n_clusters)
{
    float min_sosq = FLT_MAX;
    int m_idx = -1;

    for (int i = 0; i < n_clusters; i++) {
        float sosq = sumOfSquare(means[i], point, n_dims);
        if (min_sosq > sosq) {
            min_sosq = sosq;
            m_idx = i;
        }
    }

    return m_idx;
}

