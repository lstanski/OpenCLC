#ifndef DH
#define DH

double lineLength(double x, double y, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount);
void lineDensityRaster(float * density, int nRow, int nCol, double xMin, double yMax, double cellSize, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount);
void lineDensityRaster2(float * density, int nRow, int nCol, double xMin, double yMax, double cellSize, double r, double * lineSegX1, double * lineSegY1, double * lineSegX2, double * lineSegY2, int lineSegCount);

#endif
