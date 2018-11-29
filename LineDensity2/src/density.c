/**
 * density.c
 * Author: Ting Li <tingli3@illinois.edu>
 * Date: 08/01/2017
 */ 

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double lineLength(double x, double y, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount) {

	double lLen = 0;
	double segLen = 0;
	double x1, y1, x2, y2;
	double dis1, dis2;

	double plDis;

	//Variables for vector calcuation
	double bL;
	double aX, aY;
	double bX, bY;
	double beX, beY;

	for(int i = 0; i < lineSegCount; i++) {
		x1 = lineSegX1[i];
		y1 = lineSegY1[i];
		x2 = lineSegX2[i];
		y2 = lineSegY2[i];

		if(x1 != x2 || y1 != y2) {

			dis1 = sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
			dis2 = sqrt((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));

			segLen = 0;	
	
			if(dis1 < r) {
				if(dis2 < r) {
					segLen = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
				}
				else {
					bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
					plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	
					aX = x - x1;
					aY = y - y1;
					beX = (x2 - x1) / bL;
					beY = (y2 - y1) / bL;

					segLen = sqrt(r * r - plDis * plDis) + aX * beX + aY * beY;
				}
			}
			else {
				if(dis2 < r) {
					bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
					plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	
					aX = x - x2;
					aY = y - y2;
					beX = (x1 - x2) / bL;
					beY = (y1 - y2) / bL;

					segLen = sqrt(r * r - plDis * plDis) + aX * beX + aY * beY;
				
				}
				else {
					bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
					plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	
			
					if(plDis < r) {
						if(dis1 < dis2) {
							aX = x - x1;
							aY = y - y1;
							bX = x2 - x1;
							bY = y2 - y1;					
						}
						else {
							aX = x - x2;
							aY = y - y2;
							bX = x1 - x2;
							bY = y1 - y2;
						}
						if(aX * bX + aY * bY > 0) {
							segLen = 2 * sqrt(r * r - plDis * plDis);
						}
					}
				}
			}	

			lLen += segLen;
		}	
	}

	return lLen;
}

void lineDensityRaster(float * density, int nRow, int nCol, double xMin, double yMax, double cellSize, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount) {

	double x, y;
	double area = M_PI * r * r;

	for(int i = 0; i < nRow; i ++) {
		
		y = yMax - cellSize * (i + 0.5);
		for(int j = 0; j < nCol; j++) {
		
			x = xMin + cellSize * (j + 0.5);
		
			density[(long long)(i) * nCol + j] = (float)(lineLength(x, y, r, lineSegX1, lineSegY1, lineSegX2, lineSegY2, lineSegCount) / area);
		}
	}	
}


void lineDensityRaster2(float * density, int nRow, int nCol, double xMin, double yMax, double cellSize, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount) {

	double x, y;
	double x1, y1, x2, y2;
	int rMin, rMax;
	int cMin, cMax;
	
	double dis1, dis2;
	//Variables for vector calcuation
	
	double bL;
	double aX, aY;
	double bX, bY;
	double beX, beY;
	
	double plDis;

	double segLen;

	double area = M_PI * r * r;

	for(long long i = 0; i < (long long)(nCol) * nRow; i++) {
		density[i] = 0;
	}

	for(int i = 0; i < lineSegCount; i ++) {

		x1 = lineSegX1[i];
		y1 = lineSegY1[i];
		x2 = lineSegX2[i];
		y2 = lineSegY2[i];
		
		if(x1 != x2 || y1 != y2) {

			if(x1 > x2) {
				cMin = ceil((x2 - r - xMin) / cellSize - 0.5);
				cMax = ceil((x1 + r - xMin) / cellSize - 0.5);
			}
			else {
				cMin = ceil((x1 - r - xMin) / cellSize - 0.5);
				cMax = ceil((x2 + r - xMin) / cellSize - 0.5);
			}
			if(y1 > y2) {
				rMin = ceil((yMax - y1 - r) / cellSize - 0.5);
				rMax = ceil((yMax - y2 + r) / cellSize - 0.5);

			}
			else {
				rMin = ceil((yMax - y2 - r) / cellSize - 0.5);
				rMax = ceil((yMax - y1 + r) / cellSize - 0.5);
			}

			if(cMin < 0) cMin = 0;
			if(cMax > nCol) cMax = nCol;
			if(rMin < 0) rMin = 0;
			if(rMax > nRow) rMax = nRow;
/*
			cMin = 0;
			cMax = nCol;
			rMin = 0;
			rMax = nRow;
*/
			for(int row = rMin; row < rMax; row++) {

				y = yMax - cellSize * (row + 0.5);

				for(int c = cMin; c < cMax; c++) {
					x = xMin + cellSize * (c + 0.5);

					segLen = 0;

					dis1 = sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
					dis2 = sqrt((x2 - x) * (x2 - x) + (y2 - y) * (y2 - y));

					if(dis1 < r) {
						if(dis2 < r) {
							segLen = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
						}
						else {
							bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
							plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	
							aX = x - x1;
							aY = y - y1;
							beX = (x2 - x1) / bL;
							beY = (y2 - y1) / bL;

							segLen = sqrt(r * r - plDis * plDis) + aX * beX + aY * beY;
						}
					}
					else {
						if(dis2 < r) {
							bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
							plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	
							aX = x - x2;
							aY = y - y2;
							beX = (x1 - x2) / bL;
							beY = (y1 - y2) / bL;

							segLen = sqrt(r * r - plDis * plDis) + aX * beX + aY * beY;

						}
						else {
							bL = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
							plDis = fabs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / bL;	

							if(plDis < r) {
								if(dis1 < dis2) {
									aX = x - x1;
									aY = y - y1;
									bX = x2 - x1;
									bY = y2 - y1;					
								}
								else {
									aX = x - x2;
									aY = y - y2;
									bX = x1 - x2;
									bY = y1 - y2;
								}
								if(aX * bX + aY * bY > 0) {
									segLen = 2 * sqrt(r * r - plDis * plDis);
								}
							}
						}
					}	
					
					density[(long long)(row) * nCol + c] += (float)(segLen);
				}
			}

		}

	}

	for(long long i = 0; i < (long long)(nCol) * nRow; i++) {
		density[i] = density[i] / area;
	}

}


