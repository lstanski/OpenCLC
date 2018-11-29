#ifndef IOH
#define IOH
void registerAll();
int getLineSegCount(char * fileName);
void readPoints(char * fileName, double * x1, double * y1, double * x2, double * y2);
void readPoints(char * fileName, double * x1, double * y1, double * x2, double * y2, double &xMin, double &yMin, double &xMax, double &yMax);
void writeGeoTiff(char * fileName, float * density, int nCol, int nRow, double xMin, double yMax, double cellSize);

#endif
