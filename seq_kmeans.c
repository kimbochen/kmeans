#include <float.h>
#include <stdbool.h>
#include <stdio.h>

#include "seq_kmeans.h"

#define TOL 1e-4
#define MAX_ITER 300


void seqKMeans(
    float **points, float **means, float **clusters,
    int *cluster_sizes, int *membership,
    int n_pts, int n_feats, int n_clusters
)
{
    /*
     * points  : Data points.                                [n_pts, n_feats]
     * means   : Centroids (means) of the data points.       [n_clusters, n_feats]
     * clusters: Buffer for computing new centroids (means). [n_clusters, n_feats]
     * cluster_sizes: Size of each new cluster.              [n_clusters]
     */
    int it;
    float delta = FLT_MAX;

    // Iterate while not exceeding maximum no. of iterations and delta < tolerance
    for (it = 0; (it < MAX_ITER) && delta > TOL; it++) {
        // For every data point: find its nearest mean and add it to the corresponding cluster
        delta = 0.0;
        for (int i = 0; i < n_pts; i++) {
            int m_idx = findNearestMean(means, points[i], n_feats, n_clusters);
            if (m_idx != membership[i]) {
                membership[i] = m_idx;
                delta += 1.0;
            }
            cluster_sizes[m_idx]++;
            for (int d = 0; d < n_feats; d++) {
                clusters[m_idx][d] += points[i][d];
            }
        }
        delta /= n_pts;

        // Compute the new means
        for (int i = 0; i < n_clusters; i++) {
            for (int d = 0; d < n_feats; d++) {
                clusters[i][d] /= cluster_sizes[i];
            }
        }

        // Update the new means
        for (int i = 0; i < n_clusters; i++) {
            for (int d = 0; d < n_feats; d++) {
                means[i][d] = clusters[i][d];
            }
        }

        // Reset buffers
        for (int i = 0; i < n_clusters * n_feats; i++) {
            clusters[0][i] = 0.0;
        }
        for (int i = 0; i < n_clusters; i++) {
            cluster_sizes[i] = 0;
        }
    }

    float inertia = 0.0;
    for (int i = 0; i < n_pts; i++) {
        int m_idx = findNearestMean(means, points[i], n_feats, n_clusters);
        inertia += sumOfSquare(means[m_idx], points[i], n_feats);
    }
    printf("Converged in %d iterations. Inertia = %.3f\n", it, inertia);
}


float sumOfSquare(float *x, float *y, int n_dims)
{
    float sosq = 0.0;

    for (int d = 0; d < n_dims; d++) {
        sosq += (x[d] - y[d]) * (x[d] - y[d]);
    }

    return sosq;
}


int findNearestMean(float **means, float *point, int n_feats, int n_clusters)
{
    float min_sosq = FLT_MAX;
    int m_idx = -1;

    for (int i = 0; i < n_clusters; i++) {
        float sosq = sumOfSquare(means[i], point, n_feats);
        if (min_sosq > sosq) {
            min_sosq = sosq;
            m_idx = i;
        }
    }

    return m_idx;
}

