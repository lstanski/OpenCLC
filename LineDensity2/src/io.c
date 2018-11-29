/**
 * io.c
 * Author: Ting Li <tingli3@illinois.edu>
 * Date: 08/01/2017
 */ 

#include <stdio.h>
#include <stdlib.h>
#include <gdal.h>
#include <ogr_srs_api.h>
#include <ogr_core.h>

void registerAll() {
	GDALAllRegister();	
}


OGRSpatialReferenceH hSpatialRef;

int getLineSegCount(char * fileName) {
	GDALDatasetH hDS;

	hDS = GDALOpenEx(fileName, GDAL_OF_VECTOR, NULL, NULL, NULL);
	if( hDS == NULL ) {
		printf( "Open failed.\n" );
		exit( 1 );
	}

	OGRLayerH hLayer;
	hLayer = GDALDatasetGetLayer(hDS, 0);
	
	if(hLayer == NULL) {
		printf( "Cannot get layer\n");
		exit(1);
	}

	OGRFeatureH hFeature;
	OGR_L_ResetReading(hLayer);

	int numSeg = 0;

	int i = 0;
	while((hFeature = OGR_L_GetNextFeature(hLayer)) != NULL) {
		OGRGeometryH hGeometry;
		hGeometry = OGR_F_GetGeometryRef(hFeature);


		if(hGeometry != NULL) {		
			OGRwkbGeometryType gType = wkbFlatten(OGR_G_GetGeometryType(hGeometry));

			if(gType == wkbLineString) {
				numSeg += OGR_G_GetPointCount(hGeometry) - 1;
			}
			else if(gType == wkbMultiLineString) {
				int geoMetriCt = OGR_G_GetGeometryCount(hGeometry);

				OGRGeometryH pGeometry;
				for(int j = 0; j < geoMetriCt; j++) {
					pGeometry = OGR_G_GetGeometryRef(hGeometry, j);
					
					numSeg += OGR_G_GetPointCount(pGeometry) - 1;
				
				}
			
			}

		}

		i++;	
	}

	GDALClose(hDS);

	return numSeg;
}

void readPoints(char * fileName, double * x1, double * y1, double * x2, double * y2, double &xMin, double &yMin, double &xMax, double &yMax) {

	GDALDatasetH hDS;

	hDS = GDALOpenEx(fileName, GDAL_OF_VECTOR, NULL, NULL, NULL);
	if( hDS == NULL ) {
		printf( "Open failed.\n" );
		exit( 1 );
	}

	OGRLayerH hLayer;
	hLayer = GDALDatasetGetLayer(hDS, 0);
	
	if(hLayer == NULL) {
		printf( "Cannot get layer\n");
		exit(1);
	}

	hSpatialRef = OGR_L_GetSpatialRef(hLayer);


	OGREnvelope envelope;

	OGR_L_GetExtent(hLayer, &envelope, 1);

	xMin = envelope.MinX;
	yMin = envelope.MinY;
	xMax = envelope.MaxX;
	yMax = envelope.MaxY;

//	printf("Shape bounding box: \n %lf,\t%lf\n %lf,\t%lf\n", xMin, xMax, yMin, yMax);

	OGRFeatureH hFeature;
	OGR_L_ResetReading(hLayer);


	int pointInd = 0;

	double * xBuffer;
	double * yBuffer;


	while((hFeature = OGR_L_GetNextFeature(hLayer)) != NULL) {
		OGRGeometryH hGeometry;
		hGeometry = OGR_F_GetGeometryRef(hFeature);

		if(hGeometry != NULL) {		
			OGRwkbGeometryType gType = wkbFlatten(OGR_G_GetGeometryType(hGeometry));

			if(gType == wkbLineString) {
				int nPoints = OGR_G_GetPointCount(hGeometry);

				xBuffer = (double*) malloc(sizeof (double) * nPoints);
	                	yBuffer = (double*) malloc(sizeof (double) * nPoints);

				OGR_G_GetPoints(hGeometry, xBuffer, sizeof (double), yBuffer, sizeof (double), NULL, 0);

				for(int k = 1; k < nPoints; k++) {
					x1[pointInd] = xBuffer[k-1];
					y1[pointInd] = yBuffer[k-1];
					x2[pointInd] = xBuffer[k];
					y2[pointInd] = yBuffer[k];
					pointInd ++;
				}

				free(xBuffer);
				free(yBuffer);
			}
			else if(gType == wkbMultiLineString) {
				int geoMetriCt = OGR_G_GetGeometryCount(hGeometry);

				OGRGeometryH pGeometry;
				for(int j = 0; j < geoMetriCt; j++) {
					pGeometry = OGR_G_GetGeometryRef(hGeometry, j);
					
					int nPoints = OGR_G_GetPointCount(pGeometry);
					xBuffer = (double*) malloc(sizeof (double) * nPoints);
		                	yBuffer = (double*) malloc(sizeof (double) * nPoints);

					OGR_G_GetPoints(pGeometry, xBuffer, sizeof (double), yBuffer, sizeof (double), NULL, 0);

					for(int k = 1; k < nPoints; k++) {
						x1[pointInd] = xBuffer[k-1];
						y1[pointInd] = yBuffer[k-1];
						x2[pointInd] = xBuffer[k];
						y2[pointInd] = yBuffer[k];
						pointInd ++;
					}

					free(xBuffer);
					free(yBuffer);
			
				}	
			}
		}

	}

	GDALClose(hDS);

	return;	
}

void readPoints(char * fileName, double * x1, double * y1, double * x2, double * y2) {

	GDALDatasetH hDS;

	hDS = GDALOpenEx(fileName, GDAL_OF_VECTOR, NULL, NULL, NULL);
	if( hDS == NULL ) {
		printf( "Open failed.\n" );
		exit( 1 );
	}

	OGRLayerH hLayer;
	hLayer = GDALDatasetGetLayer(hDS, 0);
	
	if(hLayer == NULL) {
		printf( "Cannot get layer\n");
		exit(1);
	}

	hSpatialRef = OGR_L_GetSpatialRef(hLayer);

	OGRFeatureH hFeature;
	OGR_L_ResetReading(hLayer);

	int pointInd = 0;

	double * xBuffer;
	double * yBuffer;


	while((hFeature = OGR_L_GetNextFeature(hLayer)) != NULL) {
		OGRGeometryH hGeometry;
		hGeometry = OGR_F_GetGeometryRef(hFeature);

		if(hGeometry != NULL) {		
			OGRwkbGeometryType gType = wkbFlatten(OGR_G_GetGeometryType(hGeometry));

			if(gType == wkbLineString) {
				int nPoints = OGR_G_GetPointCount(hGeometry);

				xBuffer = (double*) malloc(sizeof (double) * nPoints);
	                	yBuffer = (double*) malloc(sizeof (double) * nPoints);

				OGR_G_GetPoints(hGeometry, xBuffer, sizeof (double), yBuffer, sizeof (double), NULL, 0);

				for(int k = 1; k < nPoints; k++) {
					x1[pointInd] = xBuffer[k-1];
					y1[pointInd] = yBuffer[k-1];
					x2[pointInd] = xBuffer[k];
					y2[pointInd] = yBuffer[k];
					pointInd ++;
				}

				free(xBuffer);
				free(yBuffer);
			}
			else if(gType == wkbMultiLineString) {
				int geoMetriCt = OGR_G_GetGeometryCount(hGeometry);

				OGRGeometryH pGeometry;
				for(int j = 0; j < geoMetriCt; j++) {
					pGeometry = OGR_G_GetGeometryRef(hGeometry, j);
					
					int nPoints = OGR_G_GetPointCount(pGeometry);
					xBuffer = (double*) malloc(sizeof (double) * nPoints);
		                	yBuffer = (double*) malloc(sizeof (double) * nPoints);

					OGR_G_GetPoints(pGeometry, xBuffer, sizeof (double), yBuffer, sizeof (double), NULL, 0);

					for(int k = 1; k < nPoints; k++) {
						x1[pointInd] = xBuffer[k-1];
						y1[pointInd] = yBuffer[k-1];
						x2[pointInd] = xBuffer[k];
						y2[pointInd] = yBuffer[k];
						pointInd ++;
					}

					free(xBuffer);
					free(yBuffer);
			
				}	
			}
		}

	}

	GDALClose(hDS);


	return;	
}


void writeGeoTiff(char * fileName, float * density, int nCol, int nRow, double xMin, double yMax, double cellSize) {
	
	GDALDatasetH hDstDS;
	GDALDriverH hDriver;
	GDALRasterBandH hBand;
	double adfGeoTransform[6];

	char *papszOptions[] = {"COMPRESS=LZW",NULL};
	const char *pszFormat="GTiff";

	if(NULL == (hDriver = GDALGetDriverByName(pszFormat))) {
		printf("ERROR: hDriver is null cannot output using GDAL\n");
		exit(1);
	}

	hDstDS = GDALCreate(hDriver, fileName, nCol, nRow, 1, GDT_Float32, papszOptions);

	adfGeoTransform[0] = xMin;
	adfGeoTransform[1] = cellSize;
	adfGeoTransform[2] = 0;
	adfGeoTransform[3] = yMax;
	adfGeoTransform[4] = 0;
	adfGeoTransform[5] = -cellSize;

	GDALSetGeoTransform(hDstDS,adfGeoTransform);

	if(hSpatialRef != NULL) {
		char * SRS_WKT = NULL;
		OSRExportToWkt(hSpatialRef, &SRS_WKT);
		GDALSetProjection(hDstDS, SRS_WKT);
	}

	hBand=GDALGetRasterBand(hDstDS,1);
	GDALSetRasterNoDataValue(hBand,-999);
	GDALRasterIO(hBand, GF_Write, 0, 0, nCol, nRow, density, nCol, nRow, GDT_Float32, 0, 0 );
	
	GDALClose(hDstDS);
}

