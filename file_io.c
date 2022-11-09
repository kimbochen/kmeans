#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_io.h"


void readPoints(char *fname, float ***points_p, int *n_pts_p, int *n_dims_p)
{
    FILE *fin = fopen(fname, "r");
    assert(fin != NULL);

    int status = fscanf(fin, "# %d %d", n_pts_p, n_dims_p);
    assert(status == 2);

    int n_pts = *n_pts_p;
    int n_dims = *n_dims_p;

    float **points = malloc2DFloatArray(n_pts, n_dims);
    for (int i = 0; i < n_pts; i++) {
        for (int j = 0; j < n_dims; j++) {
            int s = fscanf(fin, "%f", &points[i][j]);
            assert(s == 1);
        }
    }

    fclose(fin);

    *points_p = points;
}


void initMeans(char *fname, float ***means_p, int *n_clusters_p, float **points, int n_pts, int n_dims)
{
    FILE *fin = fopen(fname, "r");
    assert(fin != NULL);

    int status = fscanf(fin, "# %d", n_clusters_p);
    assert(status == 1);

    int idx;
    int n_clusters = *n_clusters_p;
    float **means = malloc2DFloatArray(n_clusters, n_dims);

    for (int i = 0; i < n_clusters; i++) {
        int s = fscanf(fin, "%d", &idx);
        assert(s == 1);

        for (int j = 0; j < n_dims; j++) {
            means[i][j] = points[idx][j];
        }
    }

    fclose(fin);

    *means_p = means;
}


float** malloc2DFloatArray(int rows, int cols)
{
    float **array = (float**) malloc(rows * sizeof(float*));

    array[0] = (float*) malloc(rows * cols * sizeof(float));

    for (int i = 1; i < rows; i++) {
        array[i] = array[0] + cols * i;
    }

    return array;
}


void free2DFloatArray(float **array)
{
    assert(array != NULL);
    assert(array[0] != NULL);

    free(array[0]);
    free(array);
}


void print2DFloatArray(float **array, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.8f ", array[i][j]);
        }
        puts("");
    }
}
