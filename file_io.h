#ifndef FILE_IO_H
#define FILE_IO_H

void readPoints(char*, float***, int*, int*);

void initMeans(char*, float***, int*, float**, int, int);

float** malloc2DFloatArray(int, int);

void free2DFloatArray(float**);

void print2DFloatArray(float **, int, int);

#endif
