/**
 * LineDensity.c
 * Author: Ting Li <tingli3@illinois.edu>
 * Date: 08/01/2017
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "io.h"
#include "density.h"

int main (int argc, char ** argv) {

	double cellSize;
	double r;
	double xMin, yMin, xMax, yMax;
	double xMinData, yMinData, xMaxData, yMaxData;
	int nRow, nCol;

	if(argc != 9 && argc != 5) {

		printf("ERROR: incorrect number of input parameters.\n");
		printf("To Run: LineDensity InputShapefile OutputGeoTiff OutputCellSize SearchRadius [xMin yMin xMax yMax]\n");	
		exit(1);
	}

	cellSize = atof(argv[3]);
	r = atof(argv[4]);

	nCol = ceil((xMax - xMin) / cellSize);
	nRow = ceil((yMax - yMin) / cellSize);

	xMax = xMin + nCol * cellSize;
	yMax = yMin + nRow * cellSize;

	registerAll();
	int nLineSeg = getLineSegCount(argv[1]);
	printf("%d line segments\n", nLineSeg);

	double * x1;
	double * y1;
	double * x2;
	double * y2;

	if(NULL == (x1 = (double *) malloc (sizeof(double) * nLineSeg))) {
		printf("Out of memory when allocating x1\n");
		exit(1);
	}
	if(NULL == (y1 = (double *) malloc (sizeof(double) * nLineSeg))) {
		printf("Out of memory when allocating x1\n");
		exit(1);
	}
	if(NULL == (x2 = (double *) malloc (sizeof(double) * nLineSeg))) {
		printf("Out of memory when allocating x1\n");
		exit(1);
	}
	if(NULL == (y2 = (double *) malloc (sizeof(double) * nLineSeg))) {
		printf("Out of memory when allocating x1\n");
		exit(1);
	}

	//	readPoints(argv[1], x1, y1, x2, y2);
	readPoints(argv[1], x1, y1, x2, y2, xMinData, yMinData, xMaxData, yMaxData);

	if(argc == 9) {
		xMin = atof(argv[5]);
		yMin = atof(argv[6]);
		xMax = atof(argv[7]);
		yMax = atof(argv[8]);
	}
	else {
		xMin = xMinData;
		yMin = yMinData;
		xMax = xMaxData;
		yMax = yMaxData;
	}

	cellSize = atof(argv[3]);
	r = atof(argv[4]);

	nCol = ceil((xMax - xMin) / cellSize);
	nRow = ceil((yMax - yMin) / cellSize);

	xMax = xMin + nCol * cellSize;
	yMax = yMin + nRow * cellSize;


	float * density;
	if(NULL == (density = (float *) malloc (sizeof(float) * (long long)(nRow) * nCol))) {
		printf("Out of memory when allocating density\n");
		exit(1);
	}

	lineDensityRaster2(density, nRow, nCol, xMin, yMax, cellSize, r, x1, y1, x2, y2, nLineSeg);
	
	writeGeoTiff(argv[2], density, nCol, nRow, xMin, yMax, cellSize);


	free(density);

	free(x1);
	free(y1);
	free(x2);
	free(y2);
	return 0;
}
